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
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"

#define DEBUG_TYPE "dom-tree-builder"

namespace llvm {
namespace DomTreeBuilder {

template <typename NodePtr, bool Inverse>
struct ChildrenGetter {
  static auto Get(NodePtr N) -> decltype(reverse(children<NodePtr>(N))) {
    return reverse(children<NodePtr>(N));
  }
};

template <typename NodePtr>
struct ChildrenGetter<NodePtr, true> {
  static auto Get(NodePtr N) -> decltype(inverse_children<NodePtr>(N)) {
    return inverse_children<NodePtr>(N);
  }
};

template <typename DomTreeT>
struct SemiNCAInfo {
  using NodePtr = typename DomTreeT::NodePtr;
  using NodeT = typename DomTreeT::NodeType;
  using TreeNodePtr = DomTreeNodeBase<NodeT> *;

  // Information record used by Semi-NCA during tree construction.
  struct InfoRec {
    unsigned DFSNum = 0;
    unsigned Parent = 0;
    unsigned Semi = 0;
    NodePtr Label = nullptr;
    NodePtr IDom = nullptr;
    SmallVector<NodePtr, 2> ReverseChildren;
  };

  // Number to node mapping is 1-based. Initialize the mapping to start with
  // a dummy element.
  std::vector<NodePtr> NumToNode = {nullptr};
  DenseMap<NodePtr, InfoRec> NodeToInfo;

  void clear() {
    NumToNode = {nullptr}; // Restore to initial state with a dummy start node.
    NodeToInfo.clear();
  }

  NodePtr getIDom(NodePtr BB) const {
    auto InfoIt = NodeToInfo.find(BB);
    if (InfoIt == NodeToInfo.end()) return nullptr;

    return InfoIt->second.IDom;
  }

  TreeNodePtr getNodeForBlock(NodePtr BB, DomTreeT &DT) {
    if (TreeNodePtr Node = DT.getNode(BB)) return Node;

    // Haven't calculated this node yet?  Get or calculate the node for the
    // immediate dominator.
    NodePtr IDom = getIDom(BB);

    assert(IDom || DT.DomTreeNodes[nullptr]);
    TreeNodePtr IDomNode = getNodeForBlock(IDom, DT);

    // Add a new tree node for this NodeT, and link it as a child of
    // IDomNode
    return (DT.DomTreeNodes[BB] = IDomNode->addChild(
                llvm::make_unique<DomTreeNodeBase<NodeT>>(BB, IDomNode)))
        .get();
  }

  static bool AlwaysDescend(NodePtr, NodePtr) { return true; }

  // Custom DFS implementation which can skip nodes based on a provided
  // predicate. It also collects ReverseChildren so that we don't have to spend
  // time getting predecessors in SemiNCA.
  template <bool Inverse, typename DescendCondition>
  unsigned runDFS(NodePtr V, unsigned LastNum, DescendCondition Condition,
                  unsigned AttachToNum) {
    assert(V);
    SmallVector<NodePtr, 64> WorkList = {V};
    if (NodeToInfo.count(V) != 0) NodeToInfo[V].Parent = AttachToNum;

    while (!WorkList.empty()) {
      const NodePtr BB = WorkList.pop_back_val();
      auto &BBInfo = NodeToInfo[BB];

      // Visited nodes always have positive DFS numbers.
      if (BBInfo.DFSNum != 0) continue;
      BBInfo.DFSNum = BBInfo.Semi = ++LastNum;
      BBInfo.Label = BB;
      NumToNode.push_back(BB);

      for (const NodePtr Succ : ChildrenGetter<NodePtr, Inverse>::Get(BB)) {
        const auto SIT = NodeToInfo.find(Succ);
        // Don't visit nodes more than once but remember to collect
        // RerverseChildren.
        if (SIT != NodeToInfo.end() && SIT->second.DFSNum != 0) {
          if (Succ != BB) SIT->second.ReverseChildren.push_back(BB);
          continue;
        }

        if (!Condition(BB, Succ)) continue;

        // It's fine to add Succ to the map, because we know that it will be
        // visited later.
        auto &SuccInfo = NodeToInfo[Succ];
        WorkList.push_back(Succ);
        SuccInfo.Parent = LastNum;
        SuccInfo.ReverseChildren.push_back(BB);
      }
    }

    return LastNum;
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

  void runSemiNCA(DomTreeT &DT, const unsigned MinLevel = 0) {
    const unsigned NextDFSNum(NumToNode.size());
    // Initialize IDoms to spanning tree parents.
    for (unsigned i = 1; i < NextDFSNum; ++i) {
      const NodePtr V = NumToNode[i];
      auto &VInfo = NodeToInfo[V];
      VInfo.IDom = NumToNode[VInfo.Parent];
    }

    // Step #1: Calculate the semidominators of all vertices.
    for (unsigned i = NextDFSNum - 1; i >= 2; --i) {
      NodePtr W = NumToNode[i];
      auto &WInfo = NodeToInfo[W];

      // Initialize the semi dominator to point to the parent node.
      WInfo.Semi = WInfo.Parent;
      for (const auto &N : WInfo.ReverseChildren) {
        if (NodeToInfo.count(N) == 0)  // Skip unreachable predecessors.
          continue;

        const TreeNodePtr TN = DT.getNode(N);
        // Skip predecessors whose level is above the subtree we are processing.
        if (TN && TN->getLevel() < MinLevel)
          continue;

        unsigned SemiU = NodeToInfo[eval(N, i + 1)].Semi;
        if (SemiU < WInfo.Semi) WInfo.Semi = SemiU;
      }
    }

    // Step #2: Explicitly define the immediate dominator of each vertex.
    //          IDom[i] = NCA(SDom[i], SpanningTreeParent(i)).
    // Note that the parents were stored in IDoms and later got invalidated
    // during path compression in Eval.
    for (unsigned i = 2; i < NextDFSNum; ++i) {
      const NodePtr W = NumToNode[i];
      auto &WInfo = NodeToInfo[W];
      const unsigned SDomNum = NodeToInfo[NumToNode[WInfo.Semi]].DFSNum;
      NodePtr WIDomCandidate = WInfo.IDom;
      while (NodeToInfo[WIDomCandidate].DFSNum > SDomNum)
        WIDomCandidate = NodeToInfo[WIDomCandidate].IDom;

      WInfo.IDom = WIDomCandidate;
    }
  }

  template <typename DescendCondition>
  unsigned doFullDFSWalk(const DomTreeT &DT, DescendCondition DC) {
    unsigned Num = 0;

    if (DT.Roots.size() > 1) {
      auto &BBInfo = NodeToInfo[nullptr];
      BBInfo.DFSNum = BBInfo.Semi = ++Num;
      BBInfo.Label = nullptr;

      NumToNode.push_back(nullptr);  // NumToNode[n] = V;
    }

    if (DT.isPostDominator()) {
      for (auto *Root : DT.Roots) Num = runDFS<true>(Root, Num, DC, 1);
    } else {
      assert(DT.Roots.size() == 1);
      Num = runDFS<false>(DT.Roots[0], Num, DC, Num);
    }

    return Num;
  }

  void calculateFromScratch(DomTreeT &DT, const unsigned NumBlocks) {
    // Step #0: Number blocks in depth-first order and initialize variables used
    // in later stages of the algorithm.
    const unsigned LastDFSNum = doFullDFSWalk(DT, AlwaysDescend);

    runSemiNCA(DT);

    if (DT.Roots.empty()) return;

    // Add a node for the root.  This node might be the actual root, if there is
    // one exit block, or it may be the virtual exit (denoted by
    // (BasicBlock *)0) which postdominates all real exits if there are multiple
    // exit blocks, or an infinite loop.
    // It might be that some blocks did not get a DFS number (e.g., blocks of
    // infinite loops). In these cases an artificial exit node is required.
    const bool MultipleRoots = DT.Roots.size() > 1 || (DT.isPostDominator() &&
                                                       LastDFSNum != NumBlocks);
    NodePtr Root = !MultipleRoots ? DT.Roots[0] : nullptr;

    DT.RootNode = (DT.DomTreeNodes[Root] =
                       llvm::make_unique<DomTreeNodeBase<NodeT>>(Root, nullptr))
        .get();
    attachNewSubtree(DT, DT.RootNode);
  }

  void attachNewSubtree(DomTreeT& DT, const TreeNodePtr AttachTo) {
    // Attach the first unreachable block to AttachTo.
    NodeToInfo[NumToNode[1]].IDom = AttachTo->getBlock();
    // Loop over all of the discovered blocks in the function...
    for (size_t i = 1, e = NumToNode.size(); i != e; ++i) {
      NodePtr W = NumToNode[i];
      DEBUG(dbgs() << "\tdiscovereed a new reachable node ");
      DEBUG(PrintBlockOrNullptr(dbgs(), W));
      DEBUG(dbgs() << "\n");

      // Don't replace this with 'count', the insertion side effect is important
      if (DT.DomTreeNodes[W]) continue;  // Haven't calculated this node yet?

      NodePtr ImmDom = getIDom(W);

      // Get or calculate the node for the immediate dominator
      TreeNodePtr IDomNode = getNodeForBlock(ImmDom, DT);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      DT.DomTreeNodes[W] = IDomNode->addChild(
          llvm::make_unique<DomTreeNodeBase<NodeT>>(W, IDomNode));
    }
  }

  static void PrintBlockOrNullptr(raw_ostream &O, NodePtr Obj) {
    if (!Obj)
      O << "nullptr";
    else
      Obj->printAsOperand(O, false);
  }

  // Checks if the tree contains all reachable nodes in the input graph.
  bool verifyReachability(const DomTreeT &DT) {
    clear();
    doFullDFSWalk(DT, AlwaysDescend);

    for (auto &NodeToTN : DT.DomTreeNodes) {
      const TreeNodePtr TN = NodeToTN.second.get();
      const NodePtr BB = TN->getBlock();

      // Virtual root has a corresponding virtual CFG node.
      if (DT.isVirtualRoot(TN)) continue;

      if (NodeToInfo.count(BB) == 0) {
        errs() << "DomTree node ";
        PrintBlockOrNullptr(errs(), BB);
        errs() << " not found by DFS walk!\n";
        errs().flush();

        return false;
      }
    }

    for (const NodePtr N : NumToNode) {
      if (N && !DT.getNode(N)) {
        errs() << "CFG node ";
        PrintBlockOrNullptr(errs(), N);
        errs() << " not found in the DomTree!\n";
        errs().flush();

        return false;
      }
    }

    return true;
  }

  // Check if for every parent with a level L in the tree all of its children
  // have level L + 1.
  static bool VerifyLevels(const DomTreeT &DT) {
    for (auto &NodeToTN : DT.DomTreeNodes) {
      const TreeNodePtr TN = NodeToTN.second.get();
      const NodePtr BB = TN->getBlock();
      if (!BB) continue;

      const TreeNodePtr IDom = TN->getIDom();
      if (!IDom && TN->getLevel() != 0) {
        errs() << "Node without an IDom ";
        PrintBlockOrNullptr(errs(), BB);
        errs() << " has a nonzero level " << TN->getLevel() << "!\n";
        errs().flush();

        return false;
      }

      if (IDom && TN->getLevel() != IDom->getLevel() + 1) {
        errs() << "Node ";
        PrintBlockOrNullptr(errs(), BB);
        errs() << " has level " << TN->getLevel() << " while it's IDom ";
        PrintBlockOrNullptr(errs(), IDom->getBlock());
        errs() << " has level " << IDom->getLevel() << "!\n";
        errs().flush();

        return false;
      }
    }

    return true;
  }

  // Checks if for every edge From -> To in the graph
  //     NCD(From, To) == IDom(To) or To.
  bool verifyNCD(const DomTreeT &DT) {
    clear();
    doFullDFSWalk(DT, AlwaysDescend);

    for (auto &BlockToInfo : NodeToInfo) {
      auto &Info = BlockToInfo.second;

      const NodePtr From = NumToNode[Info.Parent];
      if (!From) continue;

      const NodePtr To = BlockToInfo.first;
      const TreeNodePtr ToTN = DT.getNode(To);
      assert(ToTN);

      const NodePtr NCD = DT.findNearestCommonDominator(From, To);
      const TreeNodePtr NCDTN = DT.getNode(NCD);
      const TreeNodePtr ToIDom = ToTN->getIDom();
      if (NCDTN != ToTN && NCDTN != ToIDom) {
        errs() << "NearestCommonDominator verification failed:\n\tNCD(From:";
        PrintBlockOrNullptr(errs(), From);
        errs() << ", To:";
        PrintBlockOrNullptr(errs(), To);
        errs() << ") = ";
        PrintBlockOrNullptr(errs(), NCD);
        errs() << ",\t (should be To or IDom[To]: ";
        PrintBlockOrNullptr(errs(), ToIDom ? ToIDom->getBlock() : nullptr);
        errs() << ")\n";
        errs().flush();

        return false;
      }
    }

    return true;
  }

  // The below routines verify the correctness of the dominator tree relative to
  // the CFG it's coming from.  A tree is a dominator tree iff it has two
  // properties, called the parent property and the sibling property.  Tarjan
  // and Lengauer prove (but don't explicitly name) the properties as part of
  // the proofs in their 1972 paper, but the proofs are mostly part of proving
  // things about semidominators and idoms, and some of them are simply asserted
  // based on even earlier papers (see, e.g., lemma 2).  Some papers refer to
  // these properties as "valid" and "co-valid".  See, e.g., "Dominators,
  // directed bipolar orders, and independent spanning trees" by Loukas
  // Georgiadis and Robert E. Tarjan, as well as "Dominator Tree Verification
  // and Vertex-Disjoint Paths " by the same authors.

  // A very simple and direct explanation of these properties can be found in
  // "An Experimental Study of Dynamic Dominators", found at
  // https://arxiv.org/abs/1604.02711

  // The easiest way to think of the parent property is that it's a requirement
  // of being a dominator.  Let's just take immediate dominators.  For PARENT to
  // be an immediate dominator of CHILD, all paths in the CFG must go through
  // PARENT before they hit CHILD.  This implies that if you were to cut PARENT
  // out of the CFG, there should be no paths to CHILD that are reachable.  If
  // there are, then you now have a path from PARENT to CHILD that goes around
  // PARENT and still reaches CHILD, which by definition, means PARENT can't be
  // a dominator of CHILD (let alone an immediate one).

  // The sibling property is similar.  It says that for each pair of sibling
  // nodes in the dominator tree (LEFT and RIGHT) , they must not dominate each
  // other.  If sibling LEFT dominated sibling RIGHT, it means there are no
  // paths in the CFG from sibling LEFT to sibling RIGHT that do not go through
  // LEFT, and thus, LEFT is really an ancestor (in the dominator tree) of
  // RIGHT, not a sibling.

  // It is possible to verify the parent and sibling properties in
  // linear time, but the algorithms are complex. Instead, we do it in a
  // straightforward N^2 and N^3 way below, using direct path reachability.


  // Checks if the tree has the parent property: if for all edges from V to W in
  // the input graph, such that V is reachable, the parent of W in the tree is
  // an ancestor of V in the tree.
  //
  // This means that if a node gets disconnected from the graph, then all of
  // the nodes it dominated previously will now become unreachable.
  bool verifyParentProperty(const DomTreeT &DT) {
    for (auto &NodeToTN : DT.DomTreeNodes) {
      const TreeNodePtr TN = NodeToTN.second.get();
      const NodePtr BB = TN->getBlock();
      if (!BB || TN->getChildren().empty()) continue;

      clear();
      doFullDFSWalk(DT, [BB](NodePtr From, NodePtr To) {
        return From != BB && To != BB;
      });

      for (TreeNodePtr Child : TN->getChildren())
        if (NodeToInfo.count(Child->getBlock()) != 0) {
          errs() << "Child ";
          PrintBlockOrNullptr(errs(), Child->getBlock());
          errs() << " reachable after its parent ";
          PrintBlockOrNullptr(errs(), BB);
          errs() << " is removed!\n";
          errs().flush();

          return false;
        }
    }

    return true;
  }

  // Check if the tree has sibling property: if a node V does not dominate a
  // node W for all siblings V and W in the tree.
  //
  // This means that if a node gets disconnected from the graph, then all of its
  // siblings will now still be reachable.
  bool verifySiblingProperty(const DomTreeT &DT) {
    for (auto &NodeToTN : DT.DomTreeNodes) {
      const TreeNodePtr TN = NodeToTN.second.get();
      const NodePtr BB = TN->getBlock();
      if (!BB || TN->getChildren().empty()) continue;

      const auto &Siblings = TN->getChildren();
      for (const TreeNodePtr N : Siblings) {
        clear();
        NodePtr BBN = N->getBlock();
        doFullDFSWalk(DT, [BBN](NodePtr From, NodePtr To) {
          return From != BBN && To != BBN;
        });

        for (const TreeNodePtr S : Siblings) {
          if (S == N) continue;

          if (NodeToInfo.count(S->getBlock()) == 0) {
            errs() << "Node ";
            PrintBlockOrNullptr(errs(), S->getBlock());
            errs() << " not reachable when its sibling ";
            PrintBlockOrNullptr(errs(), N->getBlock());
            errs() << " is removed!\n";
            errs().flush();

            return false;
          }
        }
      }
    }

    return true;
  }
};


template <class DomTreeT, class FuncT>
void Calculate(DomTreeT &DT, FuncT &F) {
  SemiNCAInfo<DomTreeT> SNCA;
  SNCA.calculateFromScratch(DT, GraphTraits<FuncT *>::size(&F));
}

template <class DomTreeT>
bool Verify(const DomTreeT &DT) {
  SemiNCAInfo<DomTreeT> SNCA;
  return SNCA.verifyReachability(DT) && SNCA.VerifyLevels(DT) &&
         SNCA.verifyNCD(DT) && SNCA.verifyParentProperty(DT) &&
         SNCA.verifySiblingProperty(DT);
}

}  // namespace DomTreeBuilder
}  // namespace llvm

#undef DEBUG_TYPE

#endif
