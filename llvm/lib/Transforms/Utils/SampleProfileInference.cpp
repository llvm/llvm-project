//===- SampleProfileInference.cpp - Adjust sample profiles in the IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a profile inference algorithm. Given an incomplete and
// possibly imprecise block counts, the algorithm reconstructs realistic block
// and edge counts that satisfy flow conservation rules, while minimally modify
// input block counts.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SampleProfileInference.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <set>

using namespace llvm;
#define DEBUG_TYPE "sample-profile-inference"

namespace {

/// A value indicating an infinite flow/capacity/weight of a block/edge.
/// Not using numeric_limits<int64_t>::max(), as the values can be summed up
/// during the execution.
static constexpr int64_t INF = ((int64_t)1) << 50;

/// The minimum-cost maximum flow algorithm.
///
/// The algorithm finds the maximum flow of minimum cost on a given (directed)
/// network using a modified version of the classical Moore-Bellman-Ford
/// approach. The algorithm applies a number of augmentation iterations in which
/// flow is sent along paths of positive capacity from the source to the sink.
/// The worst-case time complexity of the implementation is O(v(f)*m*n), where
/// where m is the number of edges, n is the number of vertices, and v(f) is the
/// value of the maximum flow. However, the observed running time on typical
/// instances is sub-quadratic, that is, o(n^2).
///
/// The input is a set of edges with specified costs and capacities, and a pair
/// of nodes (source and sink). The output is the flow along each edge of the
/// minimum total cost respecting the given edge capacities.
class MinCostMaxFlow {
public:
  // Initialize algorithm's data structures for a network of a given size.
  void initialize(uint64_t NodeCount, uint64_t SourceNode, uint64_t SinkNode) {
    Source = SourceNode;
    Target = SinkNode;

    Nodes = std::vector<Node>(NodeCount);
    Edges = std::vector<std::vector<Edge>>(NodeCount, std::vector<Edge>());
  }

  // Run the algorithm.
  int64_t run() {
    // Find an augmenting path and update the flow along the path
    size_t AugmentationIters = 0;
    while (findAugmentingPath()) {
      augmentFlowAlongPath();
      AugmentationIters++;
    }

    // Compute the total flow and its cost
    int64_t TotalCost = 0;
    int64_t TotalFlow = 0;
    for (uint64_t Src = 0; Src < Nodes.size(); Src++) {
      for (auto &Edge : Edges[Src]) {
        if (Edge.Flow > 0) {
          TotalCost += Edge.Cost * Edge.Flow;
          if (Src == Source)
            TotalFlow += Edge.Flow;
        }
      }
    }
    LLVM_DEBUG(dbgs() << "Completed profi after " << AugmentationIters
                      << " iterations with " << TotalFlow << " total flow"
                      << " of " << TotalCost << " cost\n");
    (void)TotalFlow;
    return TotalCost;
  }

  /// Adding an edge to the network with a specified capacity and a cost.
  /// Multiple edges between a pair of nodes are allowed but self-edges
  /// are not supported.
  void addEdge(uint64_t Src, uint64_t Dst, int64_t Capacity, int64_t Cost) {
    assert(Capacity > 0 && "adding an edge of zero capacity");
    assert(Src != Dst && "loop edge are not supported");

    Edge SrcEdge;
    SrcEdge.Dst = Dst;
    SrcEdge.Cost = Cost;
    SrcEdge.Capacity = Capacity;
    SrcEdge.Flow = 0;
    SrcEdge.RevEdgeIndex = Edges[Dst].size();

    Edge DstEdge;
    DstEdge.Dst = Src;
    DstEdge.Cost = -Cost;
    DstEdge.Capacity = 0;
    DstEdge.Flow = 0;
    DstEdge.RevEdgeIndex = Edges[Src].size();

    Edges[Src].push_back(SrcEdge);
    Edges[Dst].push_back(DstEdge);
  }

  /// Adding an edge to the network of infinite capacity and a given cost.
  void addEdge(uint64_t Src, uint64_t Dst, int64_t Cost) {
    addEdge(Src, Dst, INF, Cost);
  }

  /// Get the total flow from a given source node.
  /// Returns a list of pairs (target node, amount of flow to the target).
  const std::vector<std::pair<uint64_t, int64_t>> getFlow(uint64_t Src) const {
    std::vector<std::pair<uint64_t, int64_t>> Flow;
    for (auto &Edge : Edges[Src]) {
      if (Edge.Flow > 0)
        Flow.push_back(std::make_pair(Edge.Dst, Edge.Flow));
    }
    return Flow;
  }

  /// Get the total flow between a pair of nodes.
  int64_t getFlow(uint64_t Src, uint64_t Dst) const {
    int64_t Flow = 0;
    for (auto &Edge : Edges[Src]) {
      if (Edge.Dst == Dst) {
        Flow += Edge.Flow;
      }
    }
    return Flow;
  }

  /// A cost of increasing a block's count by one.
  static constexpr int64_t AuxCostInc = 10;
  /// A cost of decreasing a block's count by one.
  static constexpr int64_t AuxCostDec = 20;
  /// A cost of increasing a count of zero-weight block by one.
  static constexpr int64_t AuxCostIncZero = 11;
  /// A cost of increasing the entry block's count by one.
  static constexpr int64_t AuxCostIncEntry = 40;
  /// A cost of decreasing the entry block's count by one.
  static constexpr int64_t AuxCostDecEntry = 10;
  /// A cost of taking an unlikely jump.
  static constexpr int64_t AuxCostUnlikely = ((int64_t)1) << 20;

private:
  /// Check for existence of an augmenting path with a positive capacity.
  bool findAugmentingPath() {
    // Initialize data structures
    for (auto &Node : Nodes) {
      Node.Distance = INF;
      Node.ParentNode = uint64_t(-1);
      Node.ParentEdgeIndex = uint64_t(-1);
      Node.Taken = false;
    }

    std::queue<uint64_t> Queue;
    Queue.push(Source);
    Nodes[Source].Distance = 0;
    Nodes[Source].Taken = true;
    while (!Queue.empty()) {
      uint64_t Src = Queue.front();
      Queue.pop();
      Nodes[Src].Taken = false;
      // Although the residual network contains edges with negative costs
      // (in particular, backward edges), it can be shown that there are no
      // negative-weight cycles and the following two invariants are maintained:
      // (i) Dist[Source, V] >= 0 and (ii) Dist[V, Target] >= 0 for all nodes V,
      // where Dist is the length of the shortest path between two nodes. This
      // allows to prune the search-space of the path-finding algorithm using
      // the following early-stop criteria:
      // -- If we find a path with zero-distance from Source to Target, stop the
      //    search, as the path is the shortest since Dist[Source, Target] >= 0;
      // -- If we have Dist[Source, V] > Dist[Source, Target], then do not
      //    process node V, as it is guaranteed _not_ to be on a shortest path
      //    from Source to Target; it follows from inequalities
      //    Dist[Source, Target] >= Dist[Source, V] + Dist[V, Target]
      //                         >= Dist[Source, V]
      if (Nodes[Target].Distance == 0)
        break;
      if (Nodes[Src].Distance > Nodes[Target].Distance)
        continue;

      // Process adjacent edges
      for (uint64_t EdgeIdx = 0; EdgeIdx < Edges[Src].size(); EdgeIdx++) {
        auto &Edge = Edges[Src][EdgeIdx];
        if (Edge.Flow < Edge.Capacity) {
          uint64_t Dst = Edge.Dst;
          int64_t NewDistance = Nodes[Src].Distance + Edge.Cost;
          if (Nodes[Dst].Distance > NewDistance) {
            // Update the distance and the parent node/edge
            Nodes[Dst].Distance = NewDistance;
            Nodes[Dst].ParentNode = Src;
            Nodes[Dst].ParentEdgeIndex = EdgeIdx;
            // Add the node to the queue, if it is not there yet
            if (!Nodes[Dst].Taken) {
              Queue.push(Dst);
              Nodes[Dst].Taken = true;
            }
          }
        }
      }
    }

    return Nodes[Target].Distance != INF;
  }

  /// Update the current flow along the augmenting path.
  void augmentFlowAlongPath() {
    // Find path capacity
    int64_t PathCapacity = INF;
    uint64_t Now = Target;
    while (Now != Source) {
      uint64_t Pred = Nodes[Now].ParentNode;
      auto &Edge = Edges[Pred][Nodes[Now].ParentEdgeIndex];
      PathCapacity = std::min(PathCapacity, Edge.Capacity - Edge.Flow);
      Now = Pred;
    }

    assert(PathCapacity > 0 && "found incorrect augmenting path");

    // Update the flow along the path
    Now = Target;
    while (Now != Source) {
      uint64_t Pred = Nodes[Now].ParentNode;
      auto &Edge = Edges[Pred][Nodes[Now].ParentEdgeIndex];
      auto &RevEdge = Edges[Now][Edge.RevEdgeIndex];

      Edge.Flow += PathCapacity;
      RevEdge.Flow -= PathCapacity;

      Now = Pred;
    }
  }

  /// An node in a flow network.
  struct Node {
    /// The cost of the cheapest path from the source to the current node.
    int64_t Distance;
    /// The node preceding the current one in the path.
    uint64_t ParentNode;
    /// The index of the edge between ParentNode and the current node.
    uint64_t ParentEdgeIndex;
    /// An indicator of whether the current node is in a queue.
    bool Taken;
  };
  /// An edge in a flow network.
  struct Edge {
    /// The cost of the edge.
    int64_t Cost;
    /// The capacity of the edge.
    int64_t Capacity;
    /// The current flow on the edge.
    int64_t Flow;
    /// The destination node of the edge.
    uint64_t Dst;
    /// The index of the reverse edge between Dst and the current node.
    uint64_t RevEdgeIndex;
  };

  /// The set of network nodes.
  std::vector<Node> Nodes;
  /// The set of network edges.
  std::vector<std::vector<Edge>> Edges;
  /// Source node of the flow.
  uint64_t Source;
  /// Target (sink) node of the flow.
  uint64_t Target;
};

/// Initializing flow network for a given function.
///
/// Every block is split into three nodes that are responsible for (i) an
/// incoming flow, (ii) an outgoing flow, and (iii) penalizing an increase or
/// reduction of the block weight.
void initializeNetwork(MinCostMaxFlow &Network, FlowFunction &Func) {
  uint64_t NumBlocks = Func.Blocks.size();
  assert(NumBlocks > 1 && "Too few blocks in a function");
  LLVM_DEBUG(dbgs() << "Initializing profi for " << NumBlocks << " blocks\n");

  // Pre-process data: make sure the entry weight is at least 1
  if (Func.Blocks[Func.Entry].Weight == 0) {
    Func.Blocks[Func.Entry].Weight = 1;
  }
  // Introducing dummy source/sink pairs to allow flow circulation.
  // The nodes corresponding to blocks of Func have indicies in the range
  // [0..3 * NumBlocks); the dummy nodes are indexed by the next four values.
  uint64_t S = 3 * NumBlocks;
  uint64_t T = S + 1;
  uint64_t S1 = S + 2;
  uint64_t T1 = S + 3;

  Network.initialize(3 * NumBlocks + 4, S1, T1);

  // Create three nodes for every block of the function
  for (uint64_t B = 0; B < NumBlocks; B++) {
    auto &Block = Func.Blocks[B];
    assert((!Block.UnknownWeight || Block.Weight == 0 || Block.isEntry()) &&
           "non-zero weight of a block w/o weight except for an entry");

    // Split every block into two nodes
    uint64_t Bin = 3 * B;
    uint64_t Bout = 3 * B + 1;
    uint64_t Baux = 3 * B + 2;
    if (Block.Weight > 0) {
      Network.addEdge(S1, Bout, Block.Weight, 0);
      Network.addEdge(Bin, T1, Block.Weight, 0);
    }

    // Edges from S and to T
    assert((!Block.isEntry() || !Block.isExit()) &&
           "a block cannot be an entry and an exit");
    if (Block.isEntry()) {
      Network.addEdge(S, Bin, 0);
    } else if (Block.isExit()) {
      Network.addEdge(Bout, T, 0);
    }

    // An auxiliary node to allow increase/reduction of block counts:
    // We assume that decreasing block counts is more expensive than increasing,
    // and thus, setting separate costs here. In the future we may want to tune
    // the relative costs so as to maximize the quality of generated profiles.
    int64_t AuxCostInc = MinCostMaxFlow::AuxCostInc;
    int64_t AuxCostDec = MinCostMaxFlow::AuxCostDec;
    if (Block.UnknownWeight) {
      // Do not penalize changing weights of blocks w/o known profile count
      AuxCostInc = 0;
      AuxCostDec = 0;
    } else {
      // Increasing the count for "cold" blocks with zero initial count is more
      // expensive than for "hot" ones
      if (Block.Weight == 0) {
        AuxCostInc = MinCostMaxFlow::AuxCostIncZero;
      }
      // Modifying the count of the entry block is expensive
      if (Block.isEntry()) {
        AuxCostInc = MinCostMaxFlow::AuxCostIncEntry;
        AuxCostDec = MinCostMaxFlow::AuxCostDecEntry;
      }
    }
    // For blocks with self-edges, do not penalize a reduction of the count,
    // as all of the increase can be attributed to the self-edge
    if (Block.HasSelfEdge) {
      AuxCostDec = 0;
    }

    Network.addEdge(Bin, Baux, AuxCostInc);
    Network.addEdge(Baux, Bout, AuxCostInc);
    if (Block.Weight > 0) {
      Network.addEdge(Bout, Baux, AuxCostDec);
      Network.addEdge(Baux, Bin, AuxCostDec);
    }
  }

  // Creating edges for every jump
  for (auto &Jump : Func.Jumps) {
    uint64_t Src = Jump.Source;
    uint64_t Dst = Jump.Target;
    if (Src != Dst) {
      uint64_t SrcOut = 3 * Src + 1;
      uint64_t DstIn = 3 * Dst;
      uint64_t Cost = Jump.IsUnlikely ? MinCostMaxFlow::AuxCostUnlikely : 0;
      Network.addEdge(SrcOut, DstIn, Cost);
    }
  }

  // Make sure we have a valid flow circulation
  Network.addEdge(T, S, 0);
}

/// Extract resulting block and edge counts from the flow network.
void extractWeights(MinCostMaxFlow &Network, FlowFunction &Func) {
  uint64_t NumBlocks = Func.Blocks.size();

  // Extract resulting block counts
  for (uint64_t Src = 0; Src < NumBlocks; Src++) {
    auto &Block = Func.Blocks[Src];
    uint64_t SrcOut = 3 * Src + 1;
    int64_t Flow = 0;
    for (auto &Adj : Network.getFlow(SrcOut)) {
      uint64_t DstIn = Adj.first;
      int64_t DstFlow = Adj.second;
      bool IsAuxNode = (DstIn < 3 * NumBlocks && DstIn % 3 == 2);
      if (!IsAuxNode || Block.HasSelfEdge) {
        Flow += DstFlow;
      }
    }
    Block.Flow = Flow;
    assert(Flow >= 0 && "negative block flow");
  }

  // Extract resulting jump counts
  for (auto &Jump : Func.Jumps) {
    uint64_t Src = Jump.Source;
    uint64_t Dst = Jump.Target;
    int64_t Flow = 0;
    if (Src != Dst) {
      uint64_t SrcOut = 3 * Src + 1;
      uint64_t DstIn = 3 * Dst;
      Flow = Network.getFlow(SrcOut, DstIn);
    } else {
      uint64_t SrcOut = 3 * Src + 1;
      uint64_t SrcAux = 3 * Src + 2;
      int64_t AuxFlow = Network.getFlow(SrcOut, SrcAux);
      if (AuxFlow > 0)
        Flow = AuxFlow;
    }
    Jump.Flow = Flow;
    assert(Flow >= 0 && "negative jump flow");
  }
}

#ifndef NDEBUG
/// Verify that the computed flow values satisfy flow conservation rules
void verifyWeights(const FlowFunction &Func) {
  const uint64_t NumBlocks = Func.Blocks.size();
  auto InFlow = std::vector<uint64_t>(NumBlocks, 0);
  auto OutFlow = std::vector<uint64_t>(NumBlocks, 0);
  for (auto &Jump : Func.Jumps) {
    InFlow[Jump.Target] += Jump.Flow;
    OutFlow[Jump.Source] += Jump.Flow;
  }

  uint64_t TotalInFlow = 0;
  uint64_t TotalOutFlow = 0;
  for (uint64_t I = 0; I < NumBlocks; I++) {
    auto &Block = Func.Blocks[I];
    if (Block.isEntry()) {
      TotalInFlow += Block.Flow;
      assert(Block.Flow == OutFlow[I] && "incorrectly computed control flow");
    } else if (Block.isExit()) {
      TotalOutFlow += Block.Flow;
      assert(Block.Flow == InFlow[I] && "incorrectly computed control flow");
    } else {
      assert(Block.Flow == OutFlow[I] && "incorrectly computed control flow");
      assert(Block.Flow == InFlow[I] && "incorrectly computed control flow");
    }
  }
  assert(TotalInFlow == TotalOutFlow && "incorrectly computed control flow");
}
#endif

} // end of anonymous namespace

/// Apply the profile inference algorithm for a given flow function
void llvm::applyFlowInference(FlowFunction &Func) {
  // Create and apply an inference network model
  auto InferenceNetwork = MinCostMaxFlow();
  initializeNetwork(InferenceNetwork, Func);
  InferenceNetwork.run();

  // Extract flow values for every block and every edge
  extractWeights(InferenceNetwork, Func);

#ifndef NDEBUG
  // Verify the result
  verifyWeights(Func);
#endif
}
