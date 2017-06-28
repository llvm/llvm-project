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
/// Semi-NCA algorithm described in this dissertation:
///
///   Linear-Time Algorithms for Dominators and Related Problems
///   Loukas Georgiadis, Princeton University, November 2005, pp. 21-23:
///   ftp://ftp.cs.princeton.edu/reports/2005/737.pdf
///
/// This implements the O(n*log(n)) versions of EVAL and LINK, because it turns
/// out that the theoretically slower O(n*log(n)) implementation is actually
/// faster than the almost-linear O(n*alpha(n)) version, even for large CFGs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H
#define LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

namespace DomTreeBuilder {
// Information record used by Semi-NCA during tree construction.
template <typename NodeT>
struct SemiNCAInfo {
  using NodePtr = NodeT *;
  using DomTreeT = DominatorTreeBase<NodeT>;
  using TreeNodePtr = DomTreeNodeBase<NodeT> *;

  struct InfoRec {
    unsigned DFSNum = 0;
    unsigned Parent = 0;
    unsigned Semi = 0;
    NodePtr Label = nullptr;
    NodePtr IDom = nullptr;
  };

  DomTreeT &DT;
  std::vector<NodePtr> NumToNode;
  DenseMap<NodePtr, InfoRec> NodeToInfo;

  SemiNCAInfo(DomTreeT &DT) : DT(DT) {}

  NodePtr getIDom(NodePtr BB) const {
    auto InfoIt = NodeToInfo.find(BB);
    if (InfoIt == NodeToInfo.end()) return nullptr;

    return InfoIt->second.IDom;
  }

  TreeNodePtr getNodeForBlock(NodePtr BB) {
    if (TreeNodePtr Node = DT.getNode(BB)) return Node;

    // Haven't calculated this node yet?  Get or calculate the node for the
    // immediate dominator.
    NodePtr IDom = getIDom(BB);

    assert(IDom || DT.DomTreeNodes[nullptr]);
    TreeNodePtr IDomNode = getNodeForBlock(IDom);

    // Add a new tree node for this NodeT, and link it as a child of
    // IDomNode
    return (DT.DomTreeNodes[BB] = IDomNode->addChild(
                llvm::make_unique<DomTreeNodeBase<NodeT>>(BB, IDomNode)))
        .get();
  }

  // External storage for depth first iterator that reuses the info lookup map
  // SemiNCAInfo already has. We don't have a set, but a map instead, so we are
  // converting the one argument insert calls.
  struct df_iterator_dom_storage {
   public:
    using BaseSet = decltype(NodeToInfo);
    df_iterator_dom_storage(BaseSet &Storage) : Storage(Storage) {}

    using iterator = typename BaseSet::iterator;
    std::pair<iterator, bool> insert(NodePtr N) {
      return Storage.insert({N, InfoRec()});
    }
    void completed(NodePtr) {}

   private:
    BaseSet &Storage;
  };

  df_iterator_dom_storage getStorage() { return {NodeToInfo}; }

  unsigned runReverseDFS(NodePtr V, unsigned N) {
    auto DFStorage = getStorage();

    bool IsChildOfArtificialExit = (N != 0);
    for (auto I = idf_ext_begin(V, DFStorage), E = idf_ext_end(V, DFStorage);
         I != E; ++I) {
      NodePtr BB = *I;
      auto &BBInfo = NodeToInfo[BB];
      BBInfo.DFSNum = BBInfo.Semi = ++N;
      BBInfo.Label = BB;
      // Set the parent to the top of the visited stack.  The stack includes us,
      // and is 1 based, so we subtract to account for both of these.
      if (I.getPathLength() > 1)
        BBInfo.Parent = NodeToInfo[I.getPath(I.getPathLength() - 2)].DFSNum;
      NumToNode.push_back(BB);  // NumToNode[n] = V;

      if (IsChildOfArtificialExit)
        BBInfo.Parent = 1;

      IsChildOfArtificialExit = false;
    }
    return N;
  }

  unsigned runDFS(NodePtr V, unsigned N) {
    auto DFStorage = getStorage();

    for (auto I = df_ext_begin(V, DFStorage), E = df_ext_end(V, DFStorage);
         I != E; ++I) {
      NodePtr BB = *I;
      auto &BBInfo = NodeToInfo[BB];
      BBInfo.DFSNum = BBInfo.Semi = ++N;
      BBInfo.Label = BB;
      // Set the parent to the top of the visited stack.  The stack includes us,
      // and is 1 based, so we subtract to account for both of these.
      if (I.getPathLength() > 1)
        BBInfo.Parent = NodeToInfo[I.getPath(I.getPathLength() - 2)].DFSNum;
      NumToNode.push_back(BB);  // NumToNode[n] = V;
    }
    return N;
  }

  NodePtr eval(NodePtr VIn, unsigned LastLinked) {
    auto &VInInfo = NodeToInfo[VIn];
    if (VInInfo.DFSNum < LastLinked)
      return VIn;

    SmallVector<NodePtr, 32> Work;
    SmallPtrSet<NodePtr, 32> Visited;

    if (VInInfo.Parent >= LastLinked)
      Work.push_back(VIn);

    while (!Work.empty()) {
      NodePtr V = Work.back();
      auto &VInfo = NodeToInfo[V];
      NodePtr VAncestor = NumToNode[VInfo.Parent];

      // Process Ancestor first
      if (Visited.insert(VAncestor).second && VInfo.Parent >= LastLinked) {
        Work.push_back(VAncestor);
        continue;
      }
      Work.pop_back();

      // Update VInfo based on Ancestor info
      if (VInfo.Parent < LastLinked)
        continue;

      auto &VAInfo = NodeToInfo[VAncestor];
      NodePtr VAncestorLabel = VAInfo.Label;
      NodePtr VLabel = VInfo.Label;
      if (NodeToInfo[VAncestorLabel].Semi < NodeToInfo[VLabel].Semi)
        VInfo.Label = VAncestorLabel;
      VInfo.Parent = VAInfo.Parent;
    }

    return VInInfo.Label;
  }

  template <typename NodeType>
  void runSemiNCA(unsigned NumBlocks) {
    unsigned N = 0;
    NumToNode.push_back(nullptr);

    bool MultipleRoots = (DT.Roots.size() > 1);
    if (MultipleRoots) {
      auto &BBInfo = NodeToInfo[nullptr];
      BBInfo.DFSNum = BBInfo.Semi = ++N;
      BBInfo.Label = nullptr;

      NumToNode.push_back(nullptr); // NumToNode[n] = V;
    }

    // Step #1: Number blocks in depth-first order and initialize variables used
    // in later stages of the algorithm.
    if (DT.isPostDominator()){
      for (unsigned i = 0, e = static_cast<unsigned>(DT.Roots.size());
           i != e; ++i)
        N = runReverseDFS(DT.Roots[i], N);
    } else {
      N = runDFS(DT.Roots[0], N);
    }

    // It might be that some blocks did not get a DFS number (e.g., blocks of
    // infinite loops). In these cases an artificial exit node is required.
    MultipleRoots |= (DT.isPostDominator() && N != NumBlocks);

    // Initialize IDoms to spanning tree parents.
    for (unsigned i = 1; i <= N; ++i) {
      const NodePtr V = NumToNode[i];
      auto &VInfo = NodeToInfo[V];
      VInfo.IDom = NumToNode[VInfo.Parent];
    }

    // Step #2: Calculate the semidominators of all vertices.
    for (unsigned i = N; i >= 2; --i) {
      NodePtr W = NumToNode[i];
      auto &WInfo = NodeToInfo[W];

      // Initialize the semi dominator to point to the parent node.
      WInfo.Semi = WInfo.Parent;
      for (const auto &N : inverse_children<NodeType>(W))
        if (NodeToInfo.count(N)) {  // Only if this predecessor is reachable!
          unsigned SemiU = NodeToInfo[eval(N, i + 1)].Semi;
          if (SemiU < WInfo.Semi)
            WInfo.Semi = SemiU;
        }
    }

    // Step #3: Explicitly define the immediate dominator of each vertex.
    //          IDom[i] = NCA(SDom[i], SpanningTreeParent(i)).
    // Note that the parents were stored in IDoms and later got invalidated
    // during path compression in Eval.
    for (unsigned i = 2; i <= N; ++i) {
      const NodePtr W = NumToNode[i];
      auto &WInfo = NodeToInfo[W];
      const unsigned SDomNum = NodeToInfo[NumToNode[WInfo.Semi]].DFSNum;
      NodePtr WIDomCandidate = WInfo.IDom;
      while (NodeToInfo[WIDomCandidate].DFSNum > SDomNum)
        WIDomCandidate = NodeToInfo[WIDomCandidate].IDom;

      WInfo.IDom = WIDomCandidate;
    }

    if (DT.Roots.empty()) return;

    // Add a node for the root.  This node might be the actual root, if there is
    // one exit block, or it may be the virtual exit (denoted by
    // (BasicBlock *)0) which postdominates all real exits if there are multiple
    // exit blocks, or an infinite loop.
    NodePtr Root = !MultipleRoots ? DT.Roots[0] : nullptr;

    DT.RootNode =
        (DT.DomTreeNodes[Root] =
             llvm::make_unique<DomTreeNodeBase<NodeT>>(Root, nullptr))
            .get();

    // Loop over all of the reachable blocks in the function...
    for (unsigned i = 2; i <= N; ++i) {
      NodePtr W = NumToNode[i];

      // Don't replace this with 'count', the insertion side effect is important
      if (DT.DomTreeNodes[W])
        continue; // Haven't calculated this node yet?

      NodePtr ImmDom = getIDom(W);

      assert(ImmDom || DT.DomTreeNodes[nullptr]);

      // Get or calculate the node for the immediate dominator
      TreeNodePtr IDomNode = getNodeForBlock(ImmDom);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      DT.DomTreeNodes[W] = IDomNode->addChild(
          llvm::make_unique<DomTreeNodeBase<NodeT>>(W, IDomNode));
    }

    DT.updateDFSNumbers();
  }
};
}  // namespace DomTreeBuilder

template <class FuncT, class NodeT>
void Calculate(DominatorTreeBaseByGraphTraits<GraphTraits<NodeT>> &DT,
               FuncT &F) {
  using NodePtr = typename GraphTraits<NodeT>::NodeRef;
  static_assert(std::is_pointer<NodePtr>::value,
                "NodePtr should be a pointer type");

  DomTreeBuilder::SemiNCAInfo<typename std::remove_pointer<NodePtr>::type>
      SNCA(DT);
  SNCA.template runSemiNCA<NodeT>(GraphTraits<FuncT *>::size(&F));
}
}  // namespace llvm

#endif
