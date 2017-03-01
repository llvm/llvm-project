//===- GenericDomTreeConstruction.h - Dominator Calculation ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Generic dominator tree construction - This file provides routines to
/// construct immediate dominator information for a flow-graph based on the
/// algorithm described in this document:
///
///   A Fast Algorithm for Finding Dominators in a Flowgraph
///   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
///
/// This implements the O(n*log(n)) versions of EVAL and LINK, because it turns
/// out that the theoretically slower O(n*log(n)) implementation is actually
/// faster than the almost-linear O(n*alpha(n)) version, even for large CFGs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H
#define LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

// External storage for depth first iterator that reuses the info lookup map
// domtree already has.  We don't have a set, but a map instead, so we are
// converting the one argument insert calls.
template <class NodeRef, class InfoType> struct df_iterator_dom_storage {
public:
  typedef DenseMap<NodeRef, InfoType> BaseSet;
  df_iterator_dom_storage(BaseSet &Storage) : Storage(Storage) {}

  typedef typename BaseSet::iterator iterator;
  std::pair<iterator, bool> insert(NodeRef To) {
    auto Result = Storage.insert({To, InfoType()});

    return Result;
  }
  void completed(NodeRef) {}

private:
  BaseSet &Storage;
};

template <class GraphT>
unsigned ReverseDFSPass(DominatorTreeBaseByGraphTraits<GraphT> &DT,
                        typename GraphT::NodeRef V, unsigned N) {
  df_iterator_dom_storage<
      typename GraphT::NodeRef,
      typename DominatorTreeBaseByGraphTraits<GraphT>::InfoRec>
      DFStorage(DT.Info);
  for (auto I = idf_ext_begin(V, DFStorage), E = idf_ext_end(V, DFStorage);
       I != E; ++I) {
    typename GraphT::NodeRef BB = *I;
    auto &BBInfo = DT.Info[BB];
    BBInfo.DFSNum = BBInfo.Semi = ++N;
    BBInfo.Label = BB;
    // Set the parent to the top of the visited stack.  The stack includes us,
    // and is 1 based, so we subtract to account for both of these.
    if (I.getPathLength() > 1)
      BBInfo.Parent = DT.Info[I.getPath(I.getPathLength() - 2)].DFSNum;
    DT.Vertex.push_back(BB); // Vertex[n] = V;
  }
  return N;
}
template <class GraphT>
unsigned DFSPass(DominatorTreeBaseByGraphTraits<GraphT> &DT,
                 typename GraphT::NodeRef V, unsigned N) {
  df_iterator_dom_storage<
      typename GraphT::NodeRef,
      typename DominatorTreeBaseByGraphTraits<GraphT>::InfoRec>
      DFStorage(DT.Info);
  for (auto I = df_ext_begin(V, DFStorage), E = df_ext_end(V, DFStorage);
       I != E; ++I) {
    typename GraphT::NodeRef BB = *I;
    auto &BBInfo = DT.Info[BB];
    BBInfo.DFSNum = BBInfo.Semi = ++N;
    BBInfo.Label = BB;
    // Set the parent to the top of the visited stack.  The stack includes us,
    // and is 1 based, so we subtract to account for both of these.
    if (I.getPathLength() > 1)
      BBInfo.Parent = DT.Info[I.getPath(I.getPathLength() - 2)].DFSNum;
    DT.Vertex.push_back(BB); // Vertex[n] = V;
  }
  return N;
}

template <class GraphT>
typename GraphT::NodeRef Eval(DominatorTreeBaseByGraphTraits<GraphT> &DT,
                              typename GraphT::NodeRef VIn,
                              unsigned LastLinked) {
  auto &VInInfo = DT.Info[VIn];
  if (VInInfo.DFSNum < LastLinked)
    return VIn;

  SmallVector<typename GraphT::NodeRef, 32> Work;
  SmallPtrSet<typename GraphT::NodeRef, 32> Visited;

  if (VInInfo.Parent >= LastLinked)
    Work.push_back(VIn);

  while (!Work.empty()) {
    typename GraphT::NodeRef V = Work.back();
    auto &VInfo = DT.Info[V];
    typename GraphT::NodeRef VAncestor = DT.Vertex[VInfo.Parent];

    // Process Ancestor first
    if (Visited.insert(VAncestor).second && VInfo.Parent >= LastLinked) {
      Work.push_back(VAncestor);
      continue;
    }
    Work.pop_back();

    // Update VInfo based on Ancestor info
    if (VInfo.Parent < LastLinked)
      continue;

    auto &VAInfo = DT.Info[VAncestor];
    typename GraphT::NodeRef VAncestorLabel = VAInfo.Label;
    typename GraphT::NodeRef VLabel = VInfo.Label;
    if (DT.Info[VAncestorLabel].Semi < DT.Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Parent = VAInfo.Parent;
  }

  return VInInfo.Label;
}

template <class FuncT, class NodeT>
void Calculate(DominatorTreeBaseByGraphTraits<GraphTraits<NodeT>> &DT,
               FuncT &F) {
  typedef GraphTraits<NodeT> GraphT;
  typedef GraphTraits<FuncT *> FuncGraphT;
  static_assert(std::is_pointer<typename GraphT::NodeRef>::value,
                "NodeRef should be pointer type");
  typedef typename std::remove_pointer<typename GraphT::NodeRef>::type NodeType;

  unsigned N = 0;
  bool NeedFakeRoot = DT.isPostDominator();
  // If this is post dominators, push a fake node to start
  if (NeedFakeRoot) {
    auto &BBInfo = DT.Info[nullptr];
    BBInfo.DFSNum = BBInfo.Semi = ++N;
    BBInfo.Label = nullptr;
    DT.Vertex.push_back(nullptr); // Vertex[n] = V;
  } else {
    // The root is the entry block of the CFG
    DT.addRoot(FuncGraphT::getEntryNode(&F));
  }

  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  if (DT.isPostDominator()) {
    unsigned Total = 0;
    for (auto I : nodes(&F)) {
      ++Total;
      // If it has no *successors*, it is definitely a root.
      if (FuncGraphT::child_begin(I) == FuncGraphT::child_end(I)) {
        N = ReverseDFSPass<GraphT>(DT, I, N);
        DT.Info[I].Parent = 1;
        DT.addRoot(I);
      }
    }
    // Accounting for the virtual exit, see if we had any unreachable nodes
    if (Total + 1 != N) {
      // Make another DFS pass over all other nodes to find the unreachable
      // blocks, and find the furthest paths we'll be able to make.
      // Note that this looks N^2, but it's really 2N worst case, if every node
      // is unreachable. This is because we are still going to only visit each
      // unreachable node once, we may just visit it in two directions,
      // depending on how lucky we get.
      SmallPtrSet<NodeType *, 4> ConnectToExitBlock;
      for (auto I : nodes(&F))
        if (!DT.Info.count(I)) {
          // Find the furthest away we can get by following successors, then
          // follow them in reverse.  This gives us some reasonable answer about
          // the post-dom tree inside any infinite loop.  In particular, it
          // guarantees we get to the farthest away point along *some*
          // path. This also matches GCC behavior.  If we really wanted a
          // totally complete picture of dominance inside this infinite loop, we
          // could do it with SCC-like algorithms to find the lowest and highest
          // points in the infinite loop.  In theory, it would be nice to give
          // the canonical backedge for the loop, but it's expensive.
          auto *FurthestAway = *po_begin(I);
          ConnectToExitBlock.insert(FurthestAway);
          N = ReverseDFSPass<GraphT>(DT, FurthestAway, N);
        }
      // Finally, now everything should be visited, and anything with parent ==
      // 0 should be connected to virtual exit.
      for (auto *Node : ConnectToExitBlock) {
        auto FindResult = DT.Info.find(Node);
        assert(FindResult != DT.Info.end() &&
               "Everything should have been visited by now");
        if (FindResult->second.Parent == 0) {
          FindResult->second.Parent = 1;
          DT.addRoot(Node);
        }
      }
    }
  } else {
    N = DFSPass<GraphT>(DT, GraphTraits<FuncT *>::getEntryNode(&F), N);
  }

  // When naively implemented, the Lengauer-Tarjan algorithm requires a separate
  // bucket for each vertex. However, this is unnecessary, because each vertex
  // is only placed into a single bucket (that of its semidominator), and each
  // vertex's bucket is processed before it is added to any bucket itself.
  //
  // Instead of using a bucket per vertex, we use a single array Buckets that
  // has two purposes. Before the vertex V with preorder number i is processed,
  // Buckets[i] stores the index of the first element in V's bucket. After V's
  // bucket is processed, Buckets[i] stores the index of the next element in the
  // bucket containing V, if any.
  SmallVector<unsigned, 32> Buckets;
  Buckets.resize(N + 1);
  for (unsigned i = 1; i <= N; ++i)
    Buckets[i] = i;

  for (unsigned i = N; i >= 2; --i) {
    typename GraphT::NodeRef W = DT.Vertex[i];
    auto &WInfo = DT.Info[W];

    // Step #2: Implicitly define the immediate dominator of vertices
    for (unsigned j = i; Buckets[j] != i; j = Buckets[j]) {
      typename GraphT::NodeRef V = DT.Vertex[Buckets[j]];
      typename GraphT::NodeRef U = Eval<GraphT>(DT, V, i + 1);
      DT.IDoms[V] = DT.Info[U].Semi < i ? U : W;
    }

    // Step #3: Calculate the semidominators of all vertices

    // initialize the semi dominator to point to the parent node
    WInfo.Semi = WInfo.Parent;
    for (const auto &N : inverse_children<NodeT>(W))
      if (DT.Info.count(N)) { // Only if this predecessor is reachable!
        unsigned SemiU = DT.Info[Eval<GraphT>(DT, N, i + 1)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }

    // If V is a non-root vertex and sdom(V) = parent(V), then idom(V) is
    // necessarily parent(V). In this case, set idom(V) here and avoid placing
    // V into a bucket.
    if (WInfo.Semi == WInfo.Parent) {
      DT.IDoms[W] = DT.Vertex[WInfo.Parent];
    } else {
      Buckets[i] = Buckets[WInfo.Semi];
      Buckets[WInfo.Semi] = i;
    }
  }

  if (N >= 1) {
    typename GraphT::NodeRef Root = DT.Vertex[1];
    for (unsigned j = 1; Buckets[j] != 1; j = Buckets[j]) {
      typename GraphT::NodeRef V = DT.Vertex[Buckets[j]];
      DT.IDoms[V] = Root;
    }
  }

  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeRef W = DT.Vertex[i];
    typename GraphT::NodeRef &WIDom = DT.IDoms[W];
    if (WIDom != DT.Vertex[DT.Info[W].Semi])
      WIDom = DT.IDoms[WIDom];
  }

  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks, or
  // an infinite loop.
  typename GraphT::NodeRef Root = NeedFakeRoot ? nullptr : DT.Roots[0];

  DT.RootNode =
      (DT.DomTreeNodes[Root] =
           llvm::make_unique<DomTreeNodeBase<NodeType>>(Root, nullptr))
          .get();

  // Loop over all of the reachable blocks in the function...
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeRef W = DT.Vertex[i];

    // Don't replace this with 'count', the insertion side effect is important
    if (DT.DomTreeNodes[W])
      continue; // Haven't calculated this node yet?

    typename GraphT::NodeRef ImmDom = DT.getIDom(W);

    assert(ImmDom || DT.DomTreeNodes[nullptr]);

    // Get or calculate the node for the immediate dominator
    DomTreeNodeBase<NodeType> *IDomNode = DT.getNodeForBlock(ImmDom);

    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    DT.DomTreeNodes[W] = IDomNode->addChild(
        llvm::make_unique<DomTreeNodeBase<NodeType>>(W, IDomNode));
  }

  // Free temporary memory used to construct idom's
  DT.IDoms.clear();
  DT.Info.clear();
  DT.Vertex.clear();
  DT.Vertex.shrink_to_fit();

  DT.updateDFSNumbers();
}
}

#endif
