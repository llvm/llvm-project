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
#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DDG.h"

using namespace llvm;

#define DEBUG_TYPE "dgb"

STATISTIC(TotalGraphs, "Number of dependence graphs created.");
STATISTIC(TotalDefUseEdges, "Number of def-use edges created.");
STATISTIC(TotalMemoryEdges, "Number of memory dependence edges created.");
STATISTIC(TotalFineGrainedNodes, "Number of fine-grained nodes created.");
STATISTIC(TotalPiBlockNodes, "Number of pi-block nodes created.");
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

template <class G> void AbstractDependenceGraphBuilder<G>::createPiBlocks() {
  if (!shouldCreatePiBlocks())
    return;

  LLVM_DEBUG(dbgs() << "==== Start of Creation of Pi-Blocks ===\n");

  // The overall algorithm is as follows:
  // 1. Identify SCCs and for each SCC create a pi-block node containing all
  //    the nodes in that SCC.
  // 2. Identify incoming edges incident to the nodes inside of the SCC and
  //    reconnect them to the pi-block node.
  // 3. Identify outgoing edges from the nodes inside of the SCC to nodes
  //    outside of it and reconnect them so that the edges are coming out of the
  //    SCC node instead.

  // Adding nodes as we iterate through the SCCs cause the SCC
  // iterators to get invalidated. To prevent this invalidation, we first
  // collect a list of nodes that are part of an SCC, and then iterate over
  // those lists to create the pi-block nodes. Each element of the list is a
  // list of nodes in an SCC. Note: trivial SCCs containing a single node are
  // ignored.
  SmallVector<NodeListType, 4> ListOfSCCs;
  for (auto &SCC : make_range(scc_begin(&Graph), scc_end(&Graph))) {
    if (SCC.size() > 1)
      ListOfSCCs.emplace_back(SCC.begin(), SCC.end());
  }

  for (NodeListType &NL : ListOfSCCs) {
    LLVM_DEBUG(dbgs() << "Creating pi-block node with " << NL.size()
                      << " nodes in it.\n");

    NodeType &PiNode = createPiBlock(NL);
    ++TotalPiBlockNodes;

    // Build a set to speed up the lookup for edges whose targets
    // are inside the SCC.
    SmallPtrSet<NodeType *, 4> NodesInSCC(NL.begin(), NL.end());

    // We have the set of nodes in the SCC. We go through the set of nodes
    // that are outside of the SCC and look for edges that cross the two sets.
    for (NodeType *N : Graph) {

      // Skip the SCC node and all the nodes inside of it.
      if (*N == PiNode || NodesInSCC.count(N))
        continue;

      for (NodeType *SCCNode : NL) {

        enum Direction {
          Incoming,      // Incoming edges to the SCC
          Outgoing,      // Edges going ot of the SCC
          DirectionCount // To make the enum usable as an array index.
        };

        // Use these flags to help us avoid creating redundant edges. If there
        // are more than one edges from an outside node to inside nodes, we only
        // keep one edge from that node to the pi-block node. Similarly, if
        // there are more than one edges from inside nodes to an outside node,
        // we only keep one edge from the pi-block node to the outside node.
        // There is a flag defined for each direction (incoming vs outgoing) and
        // for each type of edge supported, using a two-dimensional boolean
        // array.
        using EdgeKind = typename EdgeType::EdgeKind;
        EnumeratedArray<bool, EdgeKind> EdgeAlreadyCreated[DirectionCount]{
            false, false};

        auto createEdgeOfKind = [this](NodeType &Src, NodeType &Dst,
                                       const EdgeKind K) {
          switch (K) {
          case EdgeKind::RegisterDefUse:
            createDefUseEdge(Src, Dst);
            break;
          case EdgeKind::MemoryDependence:
            createMemoryEdge(Src, Dst);
            break;
          case EdgeKind::Rooted:
            createRootedEdge(Src, Dst);
            break;
          default:
            llvm_unreachable("Unsupported type of edge.");
          }
        };

        auto reconnectEdges = [&](NodeType *Src, NodeType *Dst, NodeType *New,
                                  const Direction Dir) {
          if (!Src->hasEdgeTo(*Dst))
            return;
          LLVM_DEBUG(dbgs()
                     << "reconnecting("
                     << (Dir == Direction::Incoming ? "incoming)" : "outgoing)")
                     << ":\nSrc:" << *Src << "\nDst:" << *Dst
                     << "\nNew:" << *New << "\n");
          assert((Dir == Direction::Incoming || Dir == Direction::Outgoing) &&
                 "Invalid direction.");

          SmallVector<EdgeType *, 10> EL;
          Src->findEdgesTo(*Dst, EL);
          for (EdgeType *OldEdge : EL) {
            EdgeKind Kind = OldEdge->getKind();
            if (!EdgeAlreadyCreated[Dir][Kind]) {
              if (Dir == Direction::Incoming) {
                createEdgeOfKind(*Src, *New, Kind);
                LLVM_DEBUG(dbgs() << "created edge from Src to New.\n");
              } else if (Dir == Direction::Outgoing) {
                createEdgeOfKind(*New, *Dst, Kind);
                LLVM_DEBUG(dbgs() << "created edge from New to Dst.\n");
              }
              EdgeAlreadyCreated[Dir][Kind] = true;
            }
            Src->removeEdge(*OldEdge);
            destroyEdge(*OldEdge);
            LLVM_DEBUG(dbgs() << "removed old edge between Src and Dst.\n\n");
          }
        };

        // Process incoming edges incident to the pi-block node.
        reconnectEdges(N, SCCNode, &PiNode, Direction::Incoming);

        // Process edges that are coming out of the pi-block node.
        reconnectEdges(SCCNode, N, &PiNode, Direction::Outgoing);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "==== End of Creation of Pi-Blocks ===\n");
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

template <class G>
void AbstractDependenceGraphBuilder<G>::sortNodesTopologically() {

  // If we don't create pi-blocks, then we may not have a DAG.
  if (!shouldCreatePiBlocks())
    return;

  SmallVector<NodeType *, 64> NodesInPO;
  for (NodeType *N : post_order(&Graph))
    NodesInPO.push_back(N);
  
  Graph.Nodes.clear();
  for (auto &N : make_range(NodesInPO.rbegin(), NodesInPO.rend()))
    Graph.Nodes.push_back(N);
}

template class llvm::AbstractDependenceGraphBuilder<DataDependenceGraph>;
template class llvm::DependenceGraphInfo<DDGNode>;
