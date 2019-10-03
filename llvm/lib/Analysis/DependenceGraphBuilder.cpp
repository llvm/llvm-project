//===- DependenceGraphBuilder.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements common steps of the build algorithm for construction
// of dependence graphs such as DDG and PDG.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DependenceGraphBuilder.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DDG.h"

using namespace llvm;

#define DEBUG_TYPE "dgb"

STATISTIC(TotalGraphs, "Number of dependence graphs created.");
STATISTIC(TotalDefUseEdges, "Number of def-use edges created.");
STATISTIC(TotalMemoryEdges, "Number of memory dependence edges created.");
STATISTIC(TotalFineGrainedNodes, "Number of fine-grained nodes created.");
STATISTIC(TotalConfusedEdges,
          "Number of confused memory dependencies between two nodes.");
STATISTIC(TotalEdgeReversals,
          "Number of times the source and sink of dependence was reversed to "
          "expose cycles in the graph.");

using InstructionListType = SmallVector<Instruction *, 2>;

//===--------------------------------------------------------------------===//
// AbstractDependenceGraphBuilder implementation
//===--------------------------------------------------------------------===//

template <class G>
void AbstractDependenceGraphBuilder<G>::createFineGrainedNodes() {
  ++TotalGraphs;
  assert(IMap.empty() && "Expected empty instruction map at start");
  for (BasicBlock *BB : BBList)
    for (Instruction &I : *BB) {
      auto &NewNode = createFineGrainedNode(I);
      IMap.insert(std::make_pair(&I, &NewNode));
      ++TotalFineGrainedNodes;
    }
}

template <class G>
void AbstractDependenceGraphBuilder<G>::createAndConnectRootNode() {
  // Create a root node that connects to every connected component of the graph.
  // This is done to allow graph iterators to visit all the disjoint components
  // of the graph, in a single walk.
  //
  // This algorithm works by going through each node of the graph and for each
  // node N, do a DFS starting from N. A rooted edge is established between the
  // root node and N (if N is not yet visited). All the nodes reachable from N
  // are marked as visited and are skipped in the DFS of subsequent nodes.
  //
  // Note: This algorithm tries to limit the number of edges out of the root
  // node to some extent, but there may be redundant edges created depending on
  // the iteration order. For example for a graph {A -> B}, an edge from the
  // root node is added to both nodes if B is visited before A. While it does
  // not result in minimal number of edges, this approach saves compile-time
  // while keeping the number of edges in check.
  auto &RootNode = createRootNode();
  df_iterator_default_set<const NodeType *, 4> Visited;
  for (auto *N : Graph) {
    if (*N == RootNode)
      continue;
    for (auto I : depth_first_ext(N, Visited))
      if (I == N)
        createRootedEdge(RootNode, *N);
  }
}

template <class G> void AbstractDependenceGraphBuilder<G>::createDefUseEdges() {
  for (NodeType *N : Graph) {
    InstructionListType SrcIList;
    N->collectInstructions([](const Instruction *I) { return true; }, SrcIList);

    // Use a set to mark the targets that we link to N, so we don't add
    // duplicate def-use edges when more than one instruction in a target node
    // use results of instructions that are contained in N.
    SmallPtrSet<NodeType *, 4> VisitedTargets;

    for (Instruction *II : SrcIList) {
      for (User *U : II->users()) {
        Instruction *UI = dyn_cast<Instruction>(U);
        if (!UI)
          continue;
        NodeType *DstNode = nullptr;
        if (IMap.find(UI) != IMap.end())
          DstNode = IMap.find(UI)->second;

        // In the case of loops, the scope of the subgraph is all the
        // basic blocks (and instructions within them) belonging to the loop. We
        // simply ignore all the edges coming from (or going into) instructions
        // or basic blocks outside of this range.
        if (!DstNode) {
          LLVM_DEBUG(
              dbgs()
              << "skipped def-use edge since the sink" << *UI
              << " is outside the range of instructions being considered.\n");
          continue;
        }

        // Self dependencies are ignored because they are redundant and
        // uninteresting.
        if (DstNode == N) {
          LLVM_DEBUG(dbgs()
                     << "skipped def-use edge since the sink and the source ("
                     << N << ") are the same.\n");
          continue;
        }

        if (VisitedTargets.insert(DstNode).second) {
          createDefUseEdge(*N, *DstNode);
          ++TotalDefUseEdges;
        }
      }
    }
  }
}

template <class G>
void AbstractDependenceGraphBuilder<G>::createMemoryDependencyEdges() {
  using DGIterator = typename G::iterator;
  auto isMemoryAccess = [](const Instruction *I) {
    return I->mayReadOrWriteMemory();
  };
  for (DGIterator SrcIt = Graph.begin(), E = Graph.end(); SrcIt != E; ++SrcIt) {
    InstructionListType SrcIList;
    (*SrcIt)->collectInstructions(isMemoryAccess, SrcIList);
    if (SrcIList.empty())
      continue;

    for (DGIterator DstIt = SrcIt; DstIt != E; ++DstIt) {
      if (**SrcIt == **DstIt)
        continue;
      InstructionListType DstIList;
      (*DstIt)->collectInstructions(isMemoryAccess, DstIList);
      if (DstIList.empty())
        continue;
      bool ForwardEdgeCreated = false;
      bool BackwardEdgeCreated = false;
      for (Instruction *ISrc : SrcIList) {
        for (Instruction *IDst : DstIList) {
          auto D = DI.depends(ISrc, IDst, true);
          if (!D)
            continue;

          // If we have a dependence with its left-most non-'=' direction
          // being '>' we need to reverse the direction of the edge, because
          // the source of the dependence cannot occur after the sink. For
          // confused dependencies, we will create edges in both directions to
          // represent the possibility of a cycle.

          auto createConfusedEdges = [&](NodeType &Src, NodeType &Dst) {
            if (!ForwardEdgeCreated) {
              createMemoryEdge(Src, Dst);
              ++TotalMemoryEdges;
            }
            if (!BackwardEdgeCreated) {
              createMemoryEdge(Dst, Src);
              ++TotalMemoryEdges;
            }
            ForwardEdgeCreated = BackwardEdgeCreated = true;
            ++TotalConfusedEdges;
          };

          auto createForwardEdge = [&](NodeType &Src, NodeType &Dst) {
            if (!ForwardEdgeCreated) {
              createMemoryEdge(Src, Dst);
              ++TotalMemoryEdges;
            }
            ForwardEdgeCreated = true;
          };

          auto createBackwardEdge = [&](NodeType &Src, NodeType &Dst) {
            if (!BackwardEdgeCreated) {
              createMemoryEdge(Dst, Src);
              ++TotalMemoryEdges;
            }
            BackwardEdgeCreated = true;
          };

          if (D->isConfused())
            createConfusedEdges(**SrcIt, **DstIt);
          else if (D->isOrdered() && !D->isLoopIndependent()) {
            bool ReversedEdge = false;
            for (unsigned Level = 1; Level <= D->getLevels(); ++Level) {
              if (D->getDirection(Level) == Dependence::DVEntry::EQ)
                continue;
              else if (D->getDirection(Level) == Dependence::DVEntry::GT) {
                createBackwardEdge(**SrcIt, **DstIt);
                ReversedEdge = true;
                ++TotalEdgeReversals;
                break;
              } else if (D->getDirection(Level) == Dependence::DVEntry::LT)
                break;
              else {
                createConfusedEdges(**SrcIt, **DstIt);
                break;
              }
            }
            if (!ReversedEdge)
              createForwardEdge(**SrcIt, **DstIt);
          } else
            createForwardEdge(**SrcIt, **DstIt);

          // Avoid creating duplicate edges.
          if (ForwardEdgeCreated && BackwardEdgeCreated)
            break;
        }

        // If we've created edges in both directions, there is no more
        // unique edge that we can create between these two nodes, so we
        // can exit early.
        if (ForwardEdgeCreated && BackwardEdgeCreated)
          break;
      }
    }
  }
}

template class llvm::AbstractDependenceGraphBuilder<DataDependenceGraph>;
template class llvm::DependenceGraphInfo<DDGNode>;
