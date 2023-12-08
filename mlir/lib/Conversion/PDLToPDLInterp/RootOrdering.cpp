//===- RootOrdering.cpp - Optimal root ordering ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An implementation of Edmonds' optimal branching algorithm. This is a
// directed analogue of the minimum spanning tree problem for a given root.
//
//===----------------------------------------------------------------------===//

#include "RootOrdering.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <queue>
#include <utility>

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

/// Returns the cycle implied by the specified parent relation, starting at the
/// given node.
static SmallVector<Value> getCycle(const DenseMap<Value, Value> &parents,
                                   Value rep) {
  SmallVector<Value> cycle;
  Value node = rep;
  do {
    cycle.push_back(node);
    node = parents.lookup(node);
    assert(node && "got an empty value in the cycle");
  } while (node != rep);
  return cycle;
}

/// Contracts the specified cycle in the given graph in-place.
/// The parentsCost map specifies, for each node in the cycle, the lowest cost
/// among the edges entering that node. Then, the nodes in the cycle C are
/// replaced with a single node v_C (the first node in the cycle). All edges
/// (u, v) entering the cycle, v \in C, are replaced with a single edge
/// (u, v_C) with an appropriately chosen cost, and the selected node v is
/// marked in the output map actualTarget[u]. All edges (u, v) leaving the
/// cycle, u \in C, are replaced with a single edge (v_C, v), and the selected
/// node u is marked in the ouptut map actualSource[v].
static void contract(RootOrderingGraph &graph, ArrayRef<Value> cycle,
                     const DenseMap<Value, unsigned> &parentDepths,
                     DenseMap<Value, Value> &actualSource,
                     DenseMap<Value, Value> &actualTarget) {
  Value rep = cycle.front();
  DenseSet<Value> cycleSet(cycle.begin(), cycle.end());

  // Now, contract the cycle, marking the actual sources and targets.
  DenseMap<Value, RootOrderingEntry> repEntries;
  for (auto outer = graph.begin(), e = graph.end(); outer != e; ++outer) {
    Value target = outer->first;
    if (cycleSet.contains(target)) {
      // Target in the cycle => edges incoming to the cycle or within the cycle.
      unsigned parentDepth = parentDepths.lookup(target);
      for (const auto &inner : outer->second) {
        Value source = inner.first;
        // Ignore edges within the cycle.
        if (cycleSet.contains(source))
          continue;

        // Edge incoming to the cycle.
        std::pair<unsigned, unsigned> cost = inner.second.cost;
        assert(parentDepth <= cost.first && "invalid parent depth");

        // Subtract the cost of the parent within the cycle from the cost of
        // the edge incoming to the cycle. This update ensures that the cost
        // of the minimum-weight spanning arborescence of the entire graph is
        // the cost of arborescence for the contracted graph plus the cost of
        // the cycle, no matter which edge in the cycle we choose to drop.
        cost.first -= parentDepth;
        auto it = repEntries.find(source);
        if (it == repEntries.end() || it->second.cost > cost) {
          actualTarget[source] = target;
          // Do not bother populating the connector (the connector is only
          // relevant for the final traversal, not for the optimal branching).
          repEntries[source].cost = cost;
        }
      }
      // Erase the node in the cycle.
      graph.erase(outer);
    } else {
      // Target not in cycle => edges going away from or unrelated to the cycle.
      DenseMap<Value, RootOrderingEntry> &entries = outer->second;
      Value bestSource;
      std::pair<unsigned, unsigned> bestCost;
      auto inner = entries.begin(), innerE = entries.end();
      while (inner != innerE) {
        Value source = inner->first;
        if (cycleSet.contains(source)) {
          // Going-away edge => get its cost and erase it.
          if (!bestSource || bestCost > inner->second.cost) {
            bestSource = source;
            bestCost = inner->second.cost;
          }
          entries.erase(inner++);
        } else {
          ++inner;
        }
      }

      // There were going-away edges, contract them.
      if (bestSource) {
        entries[rep].cost = bestCost;
        actualSource[target] = bestSource;
      }
    }
  }

  // Store the edges to the representative.
  graph[rep] = std::move(repEntries);
}

OptimalBranching::OptimalBranching(RootOrderingGraph graph, Value root)
    : graph(std::move(graph)), root(root) {}

unsigned OptimalBranching::solve() {
  // Initialize the parents and total cost.
  parents.clear();
  parents[root] = Value();
  unsigned totalCost = 0;

  // A map that stores the cost of the optimal local choice for each node
  // in a directed cycle. This map is cleared every time we seed the search.
  DenseMap<Value, unsigned> parentDepths;
  parentDepths.reserve(graph.size());

  // Determine if the optimal local choice results in an acyclic graph. This is
  // done by computing the optimal local choice and traversing up the computed
  // parents. On success, `parents` will contain the parent of each node.
  for (const auto &outer : graph) {
    Value node = outer.first;
    if (parents.count(node)) // already visited
      continue;

    // Follow the trail of best sources until we reach an already visited node.
    // The code will assert if we cannot reach an already visited node, i.e.,
    // the graph is not strongly connected.
    parentDepths.clear();
    do {
      auto it = graph.find(node);
      assert(it != graph.end() && "the graph is not strongly connected");

      // Find the best local parent, taking into account both the depth and the
      // tie breaking rules.
      Value &bestSource = parents[node];
      std::pair<unsigned, unsigned> bestCost;
      for (const auto &inner : it->second) {
        const RootOrderingEntry &entry = inner.second;
        if (!bestSource /* initial */ || bestCost > entry.cost) {
          bestSource = inner.first;
          bestCost = entry.cost;
        }
      }
      assert(bestSource && "the graph is not strongly connected");
      parentDepths[node] = bestCost.first;
      node = bestSource;
      totalCost += bestCost.first;
    } while (!parents.count(node));

    // If we reached a non-root node, we have a cycle.
    if (parentDepths.count(node)) {
      // Determine the cycle starting at the representative node.
      SmallVector<Value> cycle = getCycle(parents, node);

      // The following maps disambiguate the source / target of the edges
      // going out of / into the cycle.
      DenseMap<Value, Value> actualSource, actualTarget;

      // Contract the cycle and recurse.
      contract(graph, cycle, parentDepths, actualSource, actualTarget);
      totalCost = solve();

      // Redirect the going-away edges.
      for (auto &p : parents)
        if (p.second == node)
          // The parent is the node representating the cycle; replace it
          // with the actual (best) source in the cycle.
          p.second = actualSource.lookup(p.first);

      // Redirect the unique incoming edge and copy the cycle.
      Value parent = parents.lookup(node);
      Value entry = actualTarget.lookup(parent);
      cycle.push_back(node); // complete the cycle
      for (size_t i = 0, e = cycle.size() - 1; i < e; ++i) {
        totalCost += parentDepths.lookup(cycle[i]);
        if (cycle[i] == entry)
          parents[cycle[i]] = parent; // break the cycle
        else
          parents[cycle[i]] = cycle[i + 1];
      }

      // `parents` has a complete solution.
      break;
    }
  }

  return totalCost;
}

OptimalBranching::EdgeList
OptimalBranching::preOrderTraversal(ArrayRef<Value> nodes) const {
  // Invert the parent mapping.
  DenseMap<Value, std::vector<Value>> children;
  for (Value node : nodes) {
    if (node != root) {
      Value parent = parents.lookup(node);
      assert(parent && "invalid parent");
      children[parent].push_back(node);
    }
  }

  // The result which simultaneously acts as a queue.
  EdgeList result;
  result.reserve(nodes.size());
  result.emplace_back(root, Value());

  // Perform a BFS, pushing into the queue.
  for (size_t i = 0; i < result.size(); ++i) {
    Value node = result[i].first;
    for (Value child : children[node])
      result.emplace_back(child, node);
  }

  return result;
}
