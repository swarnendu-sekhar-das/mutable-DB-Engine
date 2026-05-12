#!/usr/bin/env python3
"""
Generate plots for extended plan enumerator benchmark results (21-30 tables).

This script visualizes the performance of various plan enumerators including:
- DPccp, DPsubOpt, TDMinCutAGaT (traditional dynamic programming)
- TwoPhaseOptimizer (Two-Phase Optimization combining Iterative Improvement and Simulated Annealing)
- TDbasic, DPsizeOpt, DPsizeSub, DPsub (other DP variants)

The plots compare optimization time across different query topologies (star, chain, cycle, clique)
and varying numbers of relations (extended to 21-30 tables).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Read the benchmark results
csv_path = 'plan_enumerator_extended.csv'
df = pd.read_csv(csv_path)

# Extract plan enumerator name from name column
df['planner'] = df['name'].str.extract(r'single core, (.+)\)')

# Calculate mean time for each planner, topology, and size
summary = df.groupby(['experiment', 'planner', 'case'])['time'].mean().reset_index()
summary['case'] = summary['case'].astype(int)

# Get unique planners and topologies from full dataframe to ensure all are included
planners = df['planner'].unique()
topologies = summary['experiment'].unique()

# Create a plot for each topology
for topology in topologies:
    topology_data = summary[summary['experiment'] == topology]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each planner
    for planner in planners:
        planner_data = topology_data[topology_data['planner'] == planner]
        if len(planner_data) > 0:
            ax.plot(planner_data['case'], planner_data['time'],
                   marker='o', label=planner, linewidth=2, markersize=6)

    ax.set_xlabel('Number of Relations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Optimization Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Plan Enumerator Performance - {topology.capitalize()} Topology',
                 fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{topology}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated plot for {topology}")

# Create a combined plot with all topologies in subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, topology in enumerate(topologies):
    topology_data = summary[summary['experiment'] == topology]
    ax = axes[idx]

    for planner in planners:
        planner_data = topology_data[topology_data['planner'] == planner]
        if len(planner_data) > 0:
            ax.plot(planner_data['case'], planner_data['time'],
                   marker='o', label=planner, linewidth=2, markersize=5)

    ax.set_xlabel('Number of Relations', fontsize=11, fontweight='bold')
    ax.set_ylabel('Optimization Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'{topology.capitalize()}', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

plt.tight_layout()
plt.savefig('all_topologies_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated combined plot for all topologies")

# Create a comparison plot focusing on TwoPhaseOptimizer vs others
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, topology in enumerate(topologies):
    topology_data = summary[summary['experiment'] == topology]
    ax = axes[idx]

    # Plot TwoPhaseOptimizer prominently
    two_phase_data = topology_data[topology_data['planner'] == 'TwoPhaseOptimizer']
    if len(two_phase_data) > 0:
        ax.plot(two_phase_data['case'], two_phase_data['time'],
               marker='o', label='TwoPhaseOptimizer', linewidth=3,
               markersize=8, color='red', alpha=0.8)

    # Plot other planners in gray
    other_planners = [p for p in planners if p != 'TwoPhaseOptimizer']
    for planner in other_planners:
        planner_data = topology_data[topology_data['planner'] == planner]
        if len(planner_data) > 0:
            ax.plot(planner_data['case'], planner_data['time'],
                   marker='o', label=planner, linewidth=1.5,
                   markersize=4, color='gray', alpha=0.5)

    ax.set_xlabel('Number of Relations', fontsize=11, fontweight='bold')
    ax.set_ylabel('Optimization Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'{topology.capitalize()}', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('twophase_vs_others.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated TwoPhaseOptimizer comparison plot")

# Create a bar chart comparing average performance across all topologies
avg_performance = summary.groupby('planner')['time'].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(range(len(avg_performance)), avg_performance.values,
              color='steelblue', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Plan Enumerator', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Optimization Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Average Performance Across All Topologies', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(avg_performance)))
ax.set_xticklabels(avg_performance.index, rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (idx, row) in enumerate(avg_performance.items()):
    ax.text(i, row, f'{row:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('average_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated average performance bar chart")

# Create a bar chart comparing geometric mean performance across all topologies
# Filter out timeout values (10000.0 ms) for geometric mean calculation only
df_filtered = df[df['time'] < 10000.0]
def geometric_mean(x):
    return np.exp(np.log(x).mean())
geo_mean_performance = df_filtered.groupby('planner')['time'].apply(geometric_mean).sort_values(ascending=True)

# Also create a plot with all algorithms including those that timed out
all_avg_performance = df.groupby('planner')['time'].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(range(len(geo_mean_performance)), geo_mean_performance.values,
              color='coral', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Plan Enumerator', fontsize=14, fontweight='bold')
ax.set_ylabel('Geometric Mean Optimization Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Geometric Mean Performance Across All Topologies', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(geo_mean_performance)))
ax.set_xticklabels(geo_mean_performance.index, rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (idx, row) in enumerate(geo_mean_performance.items()):
    ax.text(i, row, f'{row:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('geometric_mean_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated geometric mean performance bar chart")

# Create a plot with all algorithms including those that timed out
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(range(len(all_avg_performance)), all_avg_performance.values,
              color='steelblue', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Plan Enumerator', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Optimization Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Average Performance - All Algorithms (Including Timeouts)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(all_avg_performance)))
ax.set_xticklabels(all_avg_performance.index, rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (idx, row) in enumerate(all_avg_performance.items()):
    ax.text(i, row, f'{row:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('average_performance_all_algorithms.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated average performance plot for all algorithms")

# Create geometric mean plots per topology
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, topology in enumerate(topologies):
    topology_data = df_filtered[df_filtered['experiment'] == topology]
    geo_mean_per_planner = topology_data.groupby('planner')['time'].apply(geometric_mean).sort_values(ascending=True)
    
    ax = axes[idx]
    bars = ax.bar(range(len(geo_mean_per_planner)), geo_mean_per_planner.values,
                  color='coral', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Plan Enumerator', fontsize=11, fontweight='bold')
    ax.set_ylabel('Geometric Mean Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'{topology.capitalize()} - Geometric Mean', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(geo_mean_per_planner)))
    ax.set_xticklabels(geo_mean_per_planner.index, rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (idx, row) in enumerate(geo_mean_per_planner.items()):
        ax.text(i, row, f'{row:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('geometric_mean_per_topology.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated geometric mean per topology plots")

# Create median, min, max plots for all enumerators
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Median
median_performance = df.groupby('planner')['time'].median().sort_values(ascending=True)
axes[0].bar(range(len(median_performance)), median_performance.values,
           color='skyblue', edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Plan Enumerator', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Median Time (ms)', fontsize=12, fontweight='bold')
axes[0].set_title('Median Performance', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(median_performance)))
axes[0].set_xticklabels(median_performance.index, rotation=45, ha='right')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3, axis='y')

# Min
min_performance = df.groupby('planner')['time'].min().sort_values(ascending=True)
axes[1].bar(range(len(min_performance)), min_performance.values,
           color='lightgreen', edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Plan Enumerator', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Min Time (ms)', fontsize=12, fontweight='bold')
axes[1].set_title('Minimum Performance', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(min_performance)))
axes[1].set_xticklabels(min_performance.index, rotation=45, ha='right')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3, axis='y')

# Max
max_performance = df.groupby('planner')['time'].max().sort_values(ascending=True)
axes[2].bar(range(len(max_performance)), max_performance.values,
           color='salmon', edgecolor='black', linewidth=1.5)
axes[2].set_xlabel('Plan Enumerator', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Max Time (ms)', fontsize=12, fontweight='bold')
axes[2].set_title('Maximum Performance', fontsize=14, fontweight='bold')
axes[2].set_xticks(range(len(max_performance)))
axes[2].set_xticklabels(max_performance.index, rotation=45, ha='right')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('median_min_max_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated median, min, max performance plots")

# Create a comprehensive comparison plot with all metrics
fig, ax = plt.subplots(figsize=(16, 10))

metrics_data = []
for planner in planners:
    planner_data_all = df[df['planner'] == planner]['time']
    planner_data_filtered = df_filtered[df_filtered['planner'] == planner]['time']
    if len(planner_data_all) > 0:
        geo_mean_val = geometric_mean(planner_data_filtered) if len(planner_data_filtered) > 0 else float('nan')
        metrics_data.append({
            'planner': planner,
            'mean': planner_data_all.mean(),
            'median': planner_data_all.median(),
            'geometric_mean': geo_mean_val,
            'min': planner_data_all.min(),
            'max': planner_data_all.max(),
            'std': planner_data_all.std()
        })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.set_index('planner')

x = np.arange(len(metrics_df))
width = 0.15

# Plot bars, skipping NaN values for geometric mean
ax.bar(x - 2*width, metrics_df['mean'], width, label='Arithmetic Mean', color='steelblue')
ax.bar(x - width, metrics_df['median'], width, label='Median', color='skyblue')
# Only plot geometric mean for non-NaN values
geo_mean_non_nan = metrics_df['geometric_mean'].notna()
if geo_mean_non_nan.any():
    ax.bar(x[geo_mean_non_nan], metrics_df.loc[geo_mean_non_nan, 'geometric_mean'], width, label='Geometric Mean', color='coral')
ax.bar(x + width, metrics_df['min'], width, label='Min', color='lightgreen')
ax.bar(x + 2*width, metrics_df['max'], width, label='Max', color='salmon')

ax.set_xlabel('Plan Enumerator', fontsize=14, fontweight='bold')
ax.set_ylabel('Optimization Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Comprehensive Performance Metrics Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated comprehensive metrics comparison plot")

print("\nAll plots generated successfully!")
print("Generated files:")
print("- chain_performance.png")
print("- clique_performance.png")
print("- cycle_performance.png")
print("- star_performance.png")
print("- all_topologies_performance.png")
print("- twophase_vs_others.png")
print("- average_performance.png")
print("- average_performance_all_algorithms.png")
print("- geometric_mean_performance.png")
print("- geometric_mean_per_topology.png")
print("- median_min_max_performance.png")
print("- comprehensive_metrics_comparison.png")
