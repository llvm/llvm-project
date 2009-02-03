//=== llvm/Analysis/DominatorInternals.h - Dominator Calculation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATOR_INTERNALS_H
#define LLVM_ANALYSIS_DOMINATOR_INTERNALS_H

#include "llvm/Analysis/Dominators.h"
#include "llvm/ADT/SmallPtrSet.h"

//===----------------------------------------------------------------------===//
//
// DominatorTree construction - This pass constructs immediate dominator
// information for a flow-graph based on the algorithm described in this
// document:
//
//   A Fast Algorithm for Finding Dominators in a Flowgraph
//   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
//
// This implements both the O(n*ack(n)) and the O(n*log(n)) versions of EVAL and
// LINK, but it turns out that the theoretically slower O(n*log(n))
// implementation is actually faster than the "efficient" algorithm (even for
// large CFGs) because the constant overheads are substantially smaller.  The
// lower-complexity version can be enabled with the following #define:
//
#define BALANCE_IDOM_TREE 0
//
//===----------------------------------------------------------------------===//

namespace llvm {

template<class GraphT>
unsigned DFSPass(DominatorTreeBase<typename GraphT::NodeType>& DT,
                 typename GraphT::NodeType* V, unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
  InfoRec &VInfo = DT.Info[DT.Roots[i]];
  VInfo.DFSNum = VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;
  //Info[V].Ancestor = 0;     // Ancestor[n] = 0
  //Info[V].Child = 0;        // Child[v] = 0
  VInfo.Size = 1;             // Size[v] = 1

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = DT.Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DTDFSPass(DT, *SI, N);
    }
  }
#else
  bool IsChilOfArtificialExit = (N != 0);

  std::vector<std::pair<typename GraphT::NodeType*,
                        typename GraphT::ChildIteratorType> > Worklist;
  Worklist.push_back(std::make_pair(V, GraphT::child_begin(V)));
  while (!Worklist.empty()) {
    typename GraphT::NodeType* BB = Worklist.back().first;
    typename GraphT::ChildIteratorType NextSucc = Worklist.back().second;

    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &BBInfo =
                                                                    DT.Info[BB];

    // First time we visited this BB?
    if (NextSucc == GraphT::child_begin(BB)) {
      BBInfo.DFSNum = BBInfo.Semi = ++N;
      BBInfo.Label = BB;

      DT.Vertex.push_back(BB);       // Vertex[n] = V;
      //BBInfo[V].Ancestor = 0;   // Ancestor[n] = 0
      //BBInfo[V].Child = 0;      // Child[v] = 0
      BBInfo.Size = 1;            // Size[v] = 1

      if (IsChilOfArtificialExit)
        BBInfo.Parent = 1;

      IsChilOfArtificialExit = false;
    }

    // store the DFS number of the current BB - the reference to BBInfo might
    // get invalidated when processing the successors.
    unsigned BBDFSNum = BBInfo.DFSNum;

    // If we are done with this block, remove it from the worklist.
    if (NextSucc == GraphT::child_end(BB)) {
      Worklist.pop_back();
      continue;
    }

    // Increment the successor number for the next time we get to it.
    ++Worklist.back().second;
    
    // Visit the successor next, if it isn't already visited.
    typename GraphT::NodeType* Succ = *NextSucc;

    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &SuccVInfo =
                                                                  DT.Info[Succ];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = BBDFSNum;
      Worklist.push_back(std::make_pair(Succ, GraphT::child_begin(Succ)));
    }
  }
#endif
    return N;
}

template<class GraphT>
void Compress(DominatorTreeBase<typename GraphT::NodeType>& DT,
              typename GraphT::NodeType *VIn) {
  std::vector<typename GraphT::NodeType*> Work;
  SmallPtrSet<typename GraphT::NodeType*, 32> Visited;
  typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VInVAInfo =
                                      DT.Info[DT.Vertex[DT.Info[VIn].Ancestor]];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    typename GraphT::NodeType* V = Work.back();
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VInfo =
                                                                     DT.Info[V];
    typename GraphT::NodeType* VAncestor = DT.Vertex[VInfo.Ancestor];
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VAInfo =
                                                             DT.Info[VAncestor];

    // Process Ancestor first
    if (Visited.insert(VAncestor) &&
        VAInfo.Ancestor != 0) {
      Work.push_back(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VInfo based on Ancestor info
    if (VAInfo.Ancestor == 0)
      continue;
    typename GraphT::NodeType* VAncestorLabel = VAInfo.Label;
    typename GraphT::NodeType* VLabel = VInfo.Label;
    if (DT.Info[VAncestorLabel].Semi < DT.Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Ancestor = VAInfo.Ancestor;
  }
}

template<class GraphT>
typename GraphT::NodeType* Eval(DominatorTreeBase<typename GraphT::NodeType>& DT,
                                typename GraphT::NodeType *V) {
  typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VInfo =
                                                                     DT.Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress<GraphT>(DT, V);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  Compress<GraphT>(DT, V);
  GraphT::NodeType* VLabel = VInfo.Label;

  GraphT::NodeType* VAncestorLabel = DT.Info[VInfo.Ancestor].Label;
  if (DT.Info[VAncestorLabel].Semi >= DT.Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

template<class GraphT>
void Link(DominatorTreeBase<typename GraphT::NodeType>& DT,
          unsigned DFSNumV, typename GraphT::NodeType* W,
        typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &WInfo) {
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  WInfo.Ancestor = DFSNumV;
#else
  // Lower-complexity but slower implementation
  GraphT::NodeType* WLabel = WInfo.Label;
  unsigned WLabelSemi = DT.Info[WLabel].Semi;
  GraphT::NodeType* S = W;
  InfoRec *SInfo = &DT.Info[S];

  GraphT::NodeType* SChild = SInfo->Child;
  InfoRec *SChildInfo = &DT.Info[SChild];

  while (WLabelSemi < DT.Info[SChildInfo->Label].Semi) {
    GraphT::NodeType* SChildChild = SChildInfo->Child;
    if (SInfo->Size+DT.Info[SChildChild].Size >= 2*SChildInfo->Size) {
      SChildInfo->Ancestor = S;
      SInfo->Child = SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    } else {
      SChildInfo->Size = SInfo->Size;
      S = SInfo->Ancestor = SChild;
      SInfo = SChildInfo;
      SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    }
  }

  DominatorTreeBase::InfoRec &VInfo = DT.Info[V];
  SInfo->Label = WLabel;

  assert(V != W && "The optimization here will not work in this case!");
  unsigned WSize = WInfo.Size;
  unsigned VSize = (VInfo.Size += WSize);

  if (VSize < 2*WSize)
    std::swap(S, VInfo.Child);

  while (S) {
    SInfo = &DT.Info[S];
    SInfo->Ancestor = V;
    S = SInfo->Child;
  }
#endif
}

template<class FuncT, class NodeT>
void Calculate(DominatorTreeBase<typename GraphTraits<NodeT>::NodeType>& DT,
               FuncT& F) {
  typedef GraphTraits<NodeT> GraphT;

  unsigned N = 0;
  bool MultipleRoots = (DT.Roots.size() > 1);
  if (MultipleRoots) {
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &BBInfo =
        DT.Info[NULL];
    BBInfo.DFSNum = BBInfo.Semi = ++N;
    BBInfo.Label = NULL;

    DT.Vertex.push_back(NULL);       // Vertex[n] = V;
      //BBInfo[V].Ancestor = 0;   // Ancestor[n] = 0
      //BBInfo[V].Child = 0;      // Child[v] = 0
    BBInfo.Size = 1;            // Size[v] = 1
  }

  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  for (unsigned i = 0, e = static_cast<unsigned>(DT.Roots.size());
       i != e; ++i)
    N = DFSPass<GraphT>(DT, DT.Roots[i], N);

  // it might be that some blocks did not get a DFS number (e.g., blocks of 
  // infinite loops). In these cases an artificial exit node is required.
  MultipleRoots |= (DT.isPostDominator() && N != F.size());

  for (unsigned i = N; i >= 2; --i) {
    typename GraphT::NodeType* W = DT.Vertex[i];
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &WInfo =
                                                                     DT.Info[W];

    // Step #2: Calculate the semidominators of all vertices
    bool HasChildOutsideDFS = false;

    // initialize the semi dominator to point to the parent node
    WInfo.Semi = WInfo.Parent;
    for (typename GraphTraits<Inverse<NodeT> >::ChildIteratorType CI =
         GraphTraits<Inverse<NodeT> >::child_begin(W),
         E = GraphTraits<Inverse<NodeT> >::child_end(W); CI != E; ++CI) {
      if (DT.Info.count(*CI)) {  // Only if this predecessor is reachable!
        unsigned SemiU = DT.Info[Eval<GraphT>(DT, *CI)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }
      else {
        // if the child has no DFS number it is not post-dominated by any exit, 
        // and so is the current block.
        HasChildOutsideDFS = true;
      }
    }

    // if some child has no DFS number it is not post-dominated by any exit, 
    // and so is the current block.
    if (DT.isPostDominator() && HasChildOutsideDFS)
      WInfo.Semi = 0;

    DT.Info[DT.Vertex[WInfo.Semi]].Bucket.push_back(W);

    typename GraphT::NodeType* WParent = DT.Vertex[WInfo.Parent];
    Link<GraphT>(DT, WInfo.Parent, W, WInfo);

    // Step #3: Implicitly define the immediate dominator of vertices
    std::vector<typename GraphT::NodeType*> &WParentBucket =
                                                        DT.Info[WParent].Bucket;
    while (!WParentBucket.empty()) {
      typename GraphT::NodeType* V = WParentBucket.back();
      WParentBucket.pop_back();
      typename GraphT::NodeType* U = Eval<GraphT>(DT, V);
      DT.IDoms[V] = DT.Info[U].Semi < DT.Info[V].Semi ? U : WParent;
    }
  }

  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeType* W = DT.Vertex[i];
    typename GraphT::NodeType*& WIDom = DT.IDoms[W];
    if (WIDom != DT.Vertex[DT.Info[W].Semi])
      WIDom = DT.IDoms[WIDom];
  }

  if (DT.Roots.empty()) return;

  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks, or
  // an infinite loop.
  typename GraphT::NodeType* Root = !MultipleRoots ? DT.Roots[0] : 0;

  DT.DomTreeNodes[Root] = DT.RootNode =
                        new DomTreeNodeBase<typename GraphT::NodeType>(Root, 0);

  // Loop over all of the reachable blocks in the function...
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeType* W = DT.Vertex[i];

    DomTreeNodeBase<typename GraphT::NodeType> *BBNode = DT.DomTreeNodes[W];
    if (BBNode) continue;  // Haven't calculated this node yet?

    typename GraphT::NodeType* ImmDom = DT.getIDom(W);

    assert(ImmDom || DT.DomTreeNodes[NULL]);

    // Get or calculate the node for the immediate dominator
    DomTreeNodeBase<typename GraphT::NodeType> *IDomNode =
                                                     DT.getNodeForBlock(ImmDom);

    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    DomTreeNodeBase<typename GraphT::NodeType> *C =
                    new DomTreeNodeBase<typename GraphT::NodeType>(W, IDomNode);
    DT.DomTreeNodes[W] = IDomNode->addChild(C);
  }

  // Free temporary memory used to construct idom's
  DT.IDoms.clear();
  DT.Info.clear();
  std::vector<typename GraphT::NodeType*>().swap(DT.Vertex);
  
  // FIXME: This does not work on PostDomTrees.  It seems likely that this is
  // due to an error in the algorithm for post-dominators.  This really should
  // be investigated and fixed at some point.
  // DT.updateDFSNumbers();

  // Start out with the DFS numbers being invalid.  Let them be computed if
  // demanded.
  DT.DFSInfoValid = false;
}

}

#endif
