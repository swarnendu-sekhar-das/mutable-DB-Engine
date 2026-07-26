/**
 * @file TwoPhaseOptimizer.cpp
 *
 * @brief Implementation of the Two-Phase Optimization (2PO) algorithm.
 *
 * This file implements the Two-Phase Optimizer for join ordering as described in:
 * "Query Optimization by Simulated Annealing" by Ioannidis and Kang, VLDB 1987.
 *
 * Algorithm Overview:
 * The algorithm addresses the join ordering problem for large queries where exhaustive
 * dynamic programming becomes intractable. It uses a two-phase randomized search:
 *
 * Phase 1 - Iterative Improvement (II):
 *   - Starts from multiple random join orders
 *   - Performs greedy hill-climbing to local minima
 *   - Returns the best local minimum found
 *
 * Phase 2 - Simulated Annealing (SA):
 *   - Starts from the best result of Phase 1
 *   - Uses Metropolis criterion to probabilistically accept worse states
 *   - Gradually cools down to converge to global optimum
 *
 * Key Data Structures:
 * - JoinState: Represents a complete join order as a sequence of join pairs
 * - Neighborhood generation: Commutation and associativity transformations
 *
 * @note This implementation is designed for queries with 10+ relations where
 *       exhaustive DP is computationally prohibitive (O(3^n) complexity).
 */

#include <cmath>
#include <mutable/IR/PlanTable.hpp>
#include <mutable/IR/TwoPhaseOptimizer.hpp>
#include <mutable/catalog/Catalog.hpp>

namespace m {

/* JoinState Implementation */

std::vector<JoinState>
JoinState::generate_neighbors(const QueryGraph &G) const {
  std::vector<JoinState> neighbors;

  // Apply Commutation Rule: (A ⋈ B) -> (B ⋈ A)
  /* For every join pair currently in the state, we optionally commute its
     inputs. This allows the optimizer to explore alternative physical join
     directions.*/
  for (std::size_t i = 0; i < join_pairs.size(); ++i) {
    JoinState new_state = apply_commutation(i);
    neighbors.push_back(new_state);
  }

  // Apply Associativity Rule: (A ⋈ B) ⋈ C -> A ⋈ (B ⋈ C)
  /* Scans through consecutive joins and systematically changes the
     priority of joins if they match the valid associativity pattern.*/
  for (std::size_t i = 0; i < join_pairs.size() - 1; ++i) {
    JoinState new_state = apply_associativity(i);
    neighbors.push_back(new_state);
  }

  return neighbors;
}

JoinState JoinState::apply_commutation(std::size_t idx) const {
  JoinState new_state = *this;
  if (idx < join_pairs.size()) {
    std::swap(new_state.join_pairs[idx].first,
              new_state.join_pairs[idx].second);
  }
  return new_state;
}

JoinState JoinState::apply_associativity(std::size_t idx) const {
  JoinState new_state = *this;

  if (idx + 1 < join_pairs.size()) {
    auto &[left1, right1] = new_state.join_pairs[idx];
    auto &[left2, right2] = new_state.join_pairs[idx + 1];

    if (left2 == (left1 | right1)) {
      /* Pattern (X ⋈ Y) ⋈ Z -> X ⋈ (Y ⋈ Z) */
      Subproblem X = left1;
      Subproblem Y = right1;
      Subproblem Z = right2;
      new_state.join_pairs[idx] = {Y, Z};
      new_state.join_pairs[idx + 1] = {X, Y | Z};
    } else if (right2 == (left1 | right1)) {
      /* Pattern Z ⋈ (X ⋈ Y) -> (Z ⋈ X) ⋈ Y */
      Subproblem Z = left2;
      Subproblem X = left1;
      Subproblem Y = right1;
      new_state.join_pairs[idx] = {Z, X};
      new_state.join_pairs[idx + 1] = {Z | X, Y};
    }
  }

  return new_state;
}

namespace pe {

/*===== TwoPhaseOptimizer Implementation =====*/
/* The Two-Phase Optimizer (2PO) algorithm by Ioannidis & Kang tackles the join
   ordering problem for large queries. It operates in two sequential phases:
   Phase 1: Iterative Improvement (II) - Quickly finds a local minimum by
   hill-climbing multiple random start states. Phase 2: Simulated Annealing (SA)
   - Escapes local minima by probabilistically accepting worse states before
   "freezing". */

TwoPhaseOptimizer::~TwoPhaseOptimizer() = default;

template <typename PlanTable>
void TwoPhaseOptimizer::operator()(enumerate_tag, PlanTable &PT,
                                   const QueryGraph &G,
                                   const CostFunction &CF) const {

  auto &CE = Catalog::Get().get_database_in_use().cardinality_estimator();

  /* Initialize single relation plans */
  for (auto &ds : G.sources()) {
    Subproblem s = Subproblem::Singleton(ds->id());
    if (not PT.has_plan(s)) {
      PT[s].cost = 0;
      PT[s].model = CE.estimate_scan(G, s);
    }
  }

  /* Phase 1: Iterative Improvement */
  /* Generate an initial pool of random join trees and hill-climb to their local
     minima. The best local minimum out of these randomly seeded climbs is
     collected. */
  JoinState ii_best = iterative_improvement(G, PT, CF, CE);

  /* Phase 2: Simulated Annealing */
  /* Using the best state from Phase 1 as the seed, gently "anneal" the state
     over time. By sometimes taking worse paths initially, it escapes the local
     minimum trap of II. */
  JoinState final_state = simulated_annealing(ii_best, G, PT, CF, CE);

  /* Update plan table with final solution */
  update_plan_table(final_state, PT, G, CF, CE);
}

template <typename PlanTable>
JoinState
TwoPhaseOptimizer::iterative_improvement(const QueryGraph &G, PlanTable &PT,
                                         const CostFunction &CF,
                                         const CardinalityEstimator &CE) const {

  JoinState best_state = generate_random_state(G);
  best_state.cost = compute_state_cost(best_state, G, PT, CF, CE);

  const std::size_t max_iterations = 500;
  const std::size_t max_restarts = 10;

  for (std::size_t restart = 0; restart < max_restarts; ++restart) {
    /* Seed a fresh random plan (a full, randomized left-deep/bushy sequence of
       joins) */
    JoinState current_state = generate_random_state(G);
    current_state.cost = compute_state_cost(current_state, G, PT, CF, CE);

    bool improved = true;
    /* Hill-Climb: Keep taking the best neighbor until we hit a local minimum
       (i.e. no neighbor has a lower cost), or we hit maximum steps. */
    for (std::size_t iter = 0; improved && iter < max_iterations; ++iter) {
      improved = false;

      /* Generate the neighborhood (all states reachable by 1 commutation or 1
         associativity) */
      auto neighbors = current_state.generate_neighbors(G);

      for (const auto &neighbor : neighbors) {
        double neighbor_cost = compute_state_cost(neighbor, G, PT, CF, CE);

        if (neighbor_cost < current_state.cost) {
          current_state = neighbor;
          current_state.cost = neighbor_cost;
          improved = true;
        }
      }
    }

    if (current_state.cost < best_state.cost) {
      best_state = current_state;
    }
  }

  return best_state;
}

template <typename PlanTable>
JoinState TwoPhaseOptimizer::simulated_annealing(
    const JoinState &initial_state, const QueryGraph &G, PlanTable &PT,
    const CostFunction &CF, const CardinalityEstimator &CE) const {

  JoinState current_state = initial_state;
  JoinState best_state = initial_state;
  double temperature = 0.1; /* Starting temperature */
  const double cooling_rate = 0.95;

  while (temperature > 0.001) { /* Frozen condition */
    for (std::size_t iter = 0; iter < 50; ++iter) { /* Equilibrium iterations at the current temperature */
      auto neighbors = current_state.generate_neighbors(G);
      if (neighbors.empty())
        continue;
      /* SA picks a random neighbor, rather than the absolute best neighbor.
         This allows the optimizer to smoothly traverse the state space
         probabilistically. */
      std::uniform_int_distribution<std::size_t> dist(0, neighbors.size() - 1);
      JoinState neighbor = neighbors[dist(rng_)];
      neighbor.cost = compute_state_cost(neighbor, G, PT, CF, CE);

      if (neighbor.cost < current_state.cost) {
        /* Better states are always accepted automatically. */
        current_state = neighbor;
        if (neighbor.cost < best_state.cost) {
          best_state = neighbor;
        }
      } else {
        /* Metropolis Criterion: Accept worse states probabilistically.
           The larger the deficit (`delta`) or the lower the `temperature`,
           the less likely we are to accept the worse state. */
        double delta = neighbor.cost - current_state.cost;
        double probability = std::exp(-delta / temperature);
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

        if (prob_dist(rng_) < probability) {
          current_state = neighbor;
        }
      }
    }

    /* Cool down the system using an exponential cooling schedule.
       As temperature drops, the optimizer gets greedier, converging to the
       local minimum. */
    temperature *= cooling_rate;
  }

  return best_state;
}

template <typename PlanTable>
double TwoPhaseOptimizer::compute_state_cost(
    const JoinState &state, const QueryGraph &G, PlanTable &PT,
    const CostFunction &CF, const CardinalityEstimator &CE) const {
  double total_cost = 0.0;
  cnf::CNF condition; /* TODO: Extract correct condition dynamically if query
                         graph is not fully connected */

  /* Iterate through the explicitly defined join pairs sequentially
     The sum of individual join costs effectively serves as the "energy"
     of this state for purposes of SA/II. */
  for (const auto &[left, right] : state.join_pairs) {
    if (!PT.has_plan(left | right)) {
      /* Lazy evaluation: Update PlanTable only when visiting a new, unseen
         subproblem */
      PT.update(G, CE, CF, left, right, condition);
    }
    /* Calculate the local cost of this specific join to accurately reflect
       this state's energy, rather than relying on the globally optimal PT cost. */
    double local_cost = CF.calculate_join_cost(G, PT, CE, left, right, condition) - PT[left].cost - PT[right].cost;
    total_cost += local_cost;
  }

  return total_cost;
}

JoinState TwoPhaseOptimizer::generate_random_state(const QueryGraph &G) const {
  JoinState state;
  std::vector<Subproblem> relations;

  /* Collect all relations */
  for (std::size_t i = 0; i < G.num_sources(); ++i) {
    relations.push_back(Subproblem::Singleton(i));
  }

  /* Random shuffle the single-relation subsets to ensure a random physical
     sequence. */
  std::shuffle(relations.begin(), relations.end(), rng_);

  /* Iteratively pop two unattached subproblems, explicitly join them,
     and push the combined relation back. This uniformly generates random
     bushy/left-deep plans. */
  while (relations.size() > 1) {
    Subproblem first = relations.back();
    relations.pop_back();
    Subproblem second = relations.back();
    relations.pop_back();

    state.join_pairs.emplace_back(first, second);
    relations.push_back(first | second); /* Add joined relation back */
  }

  return state;
}

template <typename PlanTable>
void TwoPhaseOptimizer::update_plan_table(
    const JoinState &state, PlanTable &PT, const QueryGraph &G,
    const CostFunction &CF, const CardinalityEstimator &CE) const {
  cnf::CNF condition; /* TODO: Use actual join condition */

  /* This commits the final optimal subproblems derived by Two-Phase Optimizer
     backwards into Mutable's central `PlanTable` architecture so the executor
     can build it. */
  for (const auto &[left, right] : state.join_pairs) {
    PT.update(G, CE, CF, left, right, condition);
  }
}

/* Template instantiations */
template void TwoPhaseOptimizer::operator()<m::PlanTableSmallOrDense &>(
    enumerate_tag, PlanTableSmallOrDense &PT, const QueryGraph &G,
    const CostFunction &CF) const;
template void TwoPhaseOptimizer::operator()<m::PlanTableLargeAndSparse &>(
    enumerate_tag, PlanTableLargeAndSparse &PT, const QueryGraph &G,
    const CostFunction &CF) const;

/* Register TwoPhaseOptimizer */
__attribute__((constructor(203))) static void register_two_phase_optimizer() {
  Catalog &C = Catalog::Get();
  C.register_plan_enumerator(C.pool("TwoPhaseOptimizer"),
                             std::make_unique<TwoPhaseOptimizer>(),
                             "Two-Phase Optimization combining Iterative "
                             "Improvement and Simulated Annealing");
}

} /* namespace pe */
} /* namespace m */
