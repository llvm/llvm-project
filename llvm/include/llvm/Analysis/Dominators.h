//===- llvm/Analysis/Dominators.h - Dominator Info Calculation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the following classes:
//  1. DominatorTree: Represent dominators as an explicit tree structure.
//  2. DominanceFrontier: Calculate and hold the dominance frontier for a
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the
//  DominatorTree mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <map>
#include <set>

namespace llvm {

//===----------------------------------------------------------------------===//
/// DominatorBase - Base class that other, more interesting dominator analyses
/// inherit from.
///
template <class NodeT>
class DominatorBase {
protected:
  std::vector<NodeT*> Roots;
  const bool IsPostDominators;
  inline explicit DominatorBase(bool isPostDom) :
    Roots(), IsPostDominators(isPostDom) {}
public:

  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<NodeT*> &getRoots() const { return Roots; }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }
};


//===----------------------------------------------------------------------===//
// DomTreeNode - Dominator Tree Node
template<class NodeT> class DominatorTreeBase;
struct PostDominatorTree;
class MachineBasicBlock;

template <class NodeT>
class DomTreeNodeBase {
  NodeT *TheBB;
  DomTreeNodeBase<NodeT> *IDom;
  std::vector<DomTreeNodeBase<NodeT> *> Children;
  int DFSNumIn, DFSNumOut;

  template<class N> friend class DominatorTreeBase;
  friend struct PostDominatorTree;
public:
  typedef typename std::vector<DomTreeNodeBase<NodeT> *>::iterator iterator;
  typedef typename std::vector<DomTreeNodeBase<NodeT> *>::const_iterator
                   const_iterator;
  
  iterator begin()             { return Children.begin(); }
  iterator end()               { return Children.end(); }
  const_iterator begin() const { return Children.begin(); }
  const_iterator end()   const { return Children.end(); }
  
  NodeT *getBlock() const { return TheBB; }
  DomTreeNodeBase<NodeT> *getIDom() const { return IDom; }
  const std::vector<DomTreeNodeBase<NodeT>*> &getChildren() const {
    return Children;
  }

  DomTreeNodeBase(NodeT *BB, DomTreeNodeBase<NodeT> *iDom)
    : TheBB(BB), IDom(iDom), DFSNumIn(-1), DFSNumOut(-1) { }
  
  DomTreeNodeBase<NodeT> *addChild(DomTreeNodeBase<NodeT> *C) {
    Children.push_back(C);
    return C;
  }

  size_t getNumChildren() const {
    return Children.size();
  }

  void clearAllChildren() {
    Children.clear();
  }
  
  bool compare(DomTreeNodeBase<NodeT> *Other) {
    if (getNumChildren() != Other->getNumChildren())
      return true;

    SmallPtrSet<NodeT *, 4> OtherChildren;
    for(iterator I = Other->begin(), E = Other->end(); I != E; ++I) {
      NodeT *Nd = (*I)->getBlock();
      OtherChildren.insert(Nd);
    }

    for(iterator I = begin(), E = end(); I != E; ++I) {
      NodeT *N = (*I)->getBlock();
      if (OtherChildren.count(N) == 0)
        return true;
    }
    return false;
  }

  void setIDom(DomTreeNodeBase<NodeT> *NewIDom) {
    assert(IDom && "No immediate dominator?");
    if (IDom != NewIDom) {
      typename std::vector<DomTreeNodeBase<NodeT>*>::iterator I =
                  std::find(IDom->Children.begin(), IDom->Children.end(), this);
      assert(I != IDom->Children.end() &&
             "Not in immediate dominator children set!");
      // I am no longer your child...
      IDom->Children.erase(I);

      // Switch to new dominator
      IDom = NewIDom;
      IDom->Children.push_back(this);
    }
  }
  
  /// getDFSNumIn/getDFSNumOut - These are an internal implementation detail, do
  /// not call them.
  unsigned getDFSNumIn() const { return DFSNumIn; }
  unsigned getDFSNumOut() const { return DFSNumOut; }
private:
  // Return true if this node is dominated by other. Use this only if DFS info
  // is valid.
  bool DominatedBy(const DomTreeNodeBase<NodeT> *other) const {
    return this->DFSNumIn >= other->DFSNumIn &&
      this->DFSNumOut <= other->DFSNumOut;
  }
};

EXTERN_TEMPLATE_INSTANTIATION(class DomTreeNodeBase<BasicBlock>);
EXTERN_TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);

template<class NodeT>
static std::ostream &operator<<(std::ostream &o,
                                const DomTreeNodeBase<NodeT> *Node) {
  if (Node->getBlock())
    WriteAsOperand(o, Node->getBlock(), false);
  else
    o << " <<exit node>>";
  
  o << " {" << Node->getDFSNumIn() << "," << Node->getDFSNumOut() << "}";
  
  return o << "\n";
}

template<class NodeT>
static void PrintDomTree(const DomTreeNodeBase<NodeT> *N, std::ostream &o,
                         unsigned Lev) {
  o << std::string(2*Lev, ' ') << "[" << Lev << "] " << N;
  for (typename DomTreeNodeBase<NodeT>::const_iterator I = N->begin(),
       E = N->end(); I != E; ++I)
    PrintDomTree<NodeT>(*I, o, Lev+1);
}

typedef DomTreeNodeBase<BasicBlock> DomTreeNode;

//===----------------------------------------------------------------------===//
/// DominatorTree - Calculate the immediate dominator tree for a function.
///

template<class FuncT, class N>
void Calculate(DominatorTreeBase<typename GraphTraits<N>::NodeType>& DT,
               FuncT& F);

template<class NodeT>
class DominatorTreeBase : public DominatorBase<NodeT> {
protected:
  typedef DenseMap<NodeT*, DomTreeNodeBase<NodeT>*> DomTreeNodeMapType;
  DomTreeNodeMapType DomTreeNodes;
  DomTreeNodeBase<NodeT> *RootNode;

  bool DFSInfoValid;
  unsigned int SlowQueries;
  // Information record used during immediate dominators computation.
  struct InfoRec {
    unsigned DFSNum;
    unsigned Semi;
    unsigned Size;
    NodeT *Label, *Child;
    unsigned Parent, Ancestor;

    std::vector<NodeT*> Bucket;

    InfoRec() : DFSNum(0), Semi(0), Size(0), Label(0), Child(0), Parent(0),
                Ancestor(0) {}
  };

  DenseMap<NodeT*, NodeT*> IDoms;

  // Vertex - Map the DFS number to the BasicBlock*
  std::vector<NodeT*> Vertex;

  // Info - Collection of information used during the computation of idoms.
  DenseMap<NodeT*, InfoRec> Info;

  void reset() {
    for (typename DomTreeNodeMapType::iterator I = this->DomTreeNodes.begin(), 
           E = DomTreeNodes.end(); I != E; ++I)
      delete I->second;
    DomTreeNodes.clear();
    IDoms.clear();
    this->Roots.clear();
    Vertex.clear();
    RootNode = 0;
  }
  
  // NewBB is split and now it has one successor. Update dominator tree to
  // reflect this change.
  template<class N, class GraphT>
  void Split(DominatorTreeBase<typename GraphT::NodeType>& DT,
             typename GraphT::NodeType* NewBB) {
    assert(std::distance(GraphT::child_begin(NewBB), GraphT::child_end(NewBB)) == 1
           && "NewBB should have a single successor!");
    typename GraphT::NodeType* NewBBSucc = *GraphT::child_begin(NewBB);

    std::vector<typename GraphT::NodeType*> PredBlocks;
    for (typename GraphTraits<Inverse<N> >::ChildIteratorType PI =
         GraphTraits<Inverse<N> >::child_begin(NewBB),
         PE = GraphTraits<Inverse<N> >::child_end(NewBB); PI != PE; ++PI)
      PredBlocks.push_back(*PI);  

      assert(!PredBlocks.empty() && "No predblocks??");

      // The newly inserted basic block will dominate existing basic blocks iff the
      // PredBlocks dominate all of the non-pred blocks.  If all predblocks dominate
      // the non-pred blocks, then they all must be the same block!
      //
      bool NewBBDominatesNewBBSucc = true;
      {
        typename GraphT::NodeType* OnePred = PredBlocks[0];
        size_t i = 1, e = PredBlocks.size();
        for (i = 1; !DT.isReachableFromEntry(OnePred); ++i) {
          assert(i != e && "Didn't find reachable pred?");
          OnePred = PredBlocks[i];
        }

        for (; i != e; ++i)
          if (PredBlocks[i] != OnePred && DT.isReachableFromEntry(OnePred)) {
            NewBBDominatesNewBBSucc = false;
            break;
          }

      if (NewBBDominatesNewBBSucc)
        for (typename GraphTraits<Inverse<N> >::ChildIteratorType PI =
             GraphTraits<Inverse<N> >::child_begin(NewBBSucc),
             E = GraphTraits<Inverse<N> >::child_end(NewBBSucc); PI != E; ++PI)
          if (*PI != NewBB && !DT.dominates(NewBBSucc, *PI)) {
            NewBBDominatesNewBBSucc = false;
            break;
          }
    }

    // The other scenario where the new block can dominate its successors are when
    // all predecessors of NewBBSucc that are not NewBB are dominated by NewBBSucc
    // already.
    if (!NewBBDominatesNewBBSucc) {
      NewBBDominatesNewBBSucc = true;
      for (typename GraphTraits<Inverse<N> >::ChildIteratorType PI = 
           GraphTraits<Inverse<N> >::child_begin(NewBBSucc),
           E = GraphTraits<Inverse<N> >::child_end(NewBBSucc); PI != E; ++PI)
         if (*PI != NewBB && !DT.dominates(NewBBSucc, *PI)) {
          NewBBDominatesNewBBSucc = false;
          break;
        }
    }

    // Find NewBB's immediate dominator and create new dominator tree node for
    // NewBB.
    NodeT *NewBBIDom = 0;
    unsigned i = 0;
    for (i = 0; i < PredBlocks.size(); ++i)
      if (DT.isReachableFromEntry(PredBlocks[i])) {
        NewBBIDom = PredBlocks[i];
        break;
      }
    assert(i != PredBlocks.size() && "No reachable preds?");
    for (i = i + 1; i < PredBlocks.size(); ++i) {
      if (DT.isReachableFromEntry(PredBlocks[i]))
        NewBBIDom = DT.findNearestCommonDominator(NewBBIDom, PredBlocks[i]);
    }
    assert(NewBBIDom && "No immediate dominator found??");

    // Create the new dominator tree node... and set the idom of NewBB.
    DomTreeNodeBase<NodeT> *NewBBNode = DT.addNewBlock(NewBB, NewBBIDom);

    // If NewBB strictly dominates other blocks, then it is now the immediate
    // dominator of NewBBSucc.  Update the dominator tree as appropriate.
    if (NewBBDominatesNewBBSucc) {
      DomTreeNodeBase<NodeT> *NewBBSuccNode = DT.getNode(NewBBSucc);
      DT.changeImmediateDominator(NewBBSuccNode, NewBBNode);
    }
  }

public:
  explicit DominatorTreeBase(bool isPostDom)
    : DominatorBase<NodeT>(isPostDom), DFSInfoValid(false), SlowQueries(0) {}
  virtual ~DominatorTreeBase() { reset(); }

  // FIXME: Should remove this
  virtual bool runOnFunction(Function &F) { return false; }

  /// compare - Return false if the other dominator tree base matches this
  /// dominator tree base. Otherwise return true.
  bool compare(DominatorTreeBase &Other) const {

    const DomTreeNodeMapType &OtherDomTreeNodes = Other.DomTreeNodes;
    if (DomTreeNodes.size() != OtherDomTreeNodes.size())
      return true;

    SmallPtrSet<const NodeT *,4> MyBBs;
    for (typename DomTreeNodeMapType::const_iterator 
           I = this->DomTreeNodes.begin(),
           E = this->DomTreeNodes.end(); I != E; ++I) {
      NodeT *BB = I->first;
      typename DomTreeNodeMapType::const_iterator OI = OtherDomTreeNodes.find(BB);
      if (OI == OtherDomTreeNodes.end())
        return true;

      DomTreeNodeBase<NodeT>* MyNd = I->second;
      DomTreeNodeBase<NodeT>* OtherNd = OI->second;
      
      if (MyNd->compare(OtherNd))
        return true;
    }

    return false;
  }

  virtual void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline DomTreeNodeBase<NodeT> *getNode(NodeT *BB) const {
    typename DomTreeNodeMapType::const_iterator I = DomTreeNodes.find(BB);
    return I != DomTreeNodes.end() ? I->second : 0;
  }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  ///
  DomTreeNodeBase<NodeT> *getRootNode() { return RootNode; }
  const DomTreeNodeBase<NodeT> *getRootNode() const { return RootNode; }

  /// properlyDominates - Returns true iff this dominates N and this != N.
  /// Note that this is not a constant time operation!
  ///
  bool properlyDominates(const DomTreeNodeBase<NodeT> *A,
                         DomTreeNodeBase<NodeT> *B) const {
    if (A == 0 || B == 0) return false;
    return dominatedBySlowTreeWalk(A, B);
  }

  inline bool properlyDominates(NodeT *A, NodeT *B) {
    return properlyDominates(getNode(A), getNode(B));
  }

  bool dominatedBySlowTreeWalk(const DomTreeNodeBase<NodeT> *A, 
                               const DomTreeNodeBase<NodeT> *B) const {
    const DomTreeNodeBase<NodeT> *IDom;
    if (A == 0 || B == 0) return false;
    while ((IDom = B->getIDom()) != 0 && IDom != A && IDom != B)
      B = IDom;   // Walk up the tree
    return IDom != 0;
  }


  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  bool isReachableFromEntry(NodeT* A) {
    assert (!this->isPostDominator() 
            && "This is not implemented for post dominators");
    return dominates(&A->getParent()->front(), A);
  }
  
  /// dominates - Returns true iff A dominates B.  Note that this is not a
  /// constant time operation!
  ///
  inline bool dominates(const DomTreeNodeBase<NodeT> *A,
                        DomTreeNodeBase<NodeT> *B) {
    if (B == A) 
      return true;  // A node trivially dominates itself.

    if (A == 0 || B == 0)
      return false;

    if (DFSInfoValid)
      return B->DominatedBy(A);

    // If we end up with too many slow queries, just update the
    // DFS numbers on the theory that we are going to keep querying.
    SlowQueries++;
    if (SlowQueries > 32) {
      updateDFSNumbers();
      return B->DominatedBy(A);
    }

    return dominatedBySlowTreeWalk(A, B);
  }

  inline bool dominates(NodeT *A, NodeT *B) {
    if (A == B) 
      return true;
    
    return dominates(getNode(A), getNode(B));
  }
  
  NodeT *getRoot() const {
    assert(this->Roots.size() == 1 && "Should always have entry node!");
    return this->Roots[0];
  }

  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  NodeT *findNearestCommonDominator(NodeT *A, NodeT *B) {

    assert (!this->isPostDominator() 
            && "This is not implemented for post dominators");
    assert (A->getParent() == B->getParent() 
            && "Two blocks are not in same function");

    // If either A or B is a entry block then it is nearest common dominator.
    NodeT &Entry  = A->getParent()->front();
    if (A == &Entry || B == &Entry)
      return &Entry;

    // If B dominates A then B is nearest common dominator.
    if (dominates(B, A))
      return B;

    // If A dominates B then A is nearest common dominator.
    if (dominates(A, B))
      return A;

    DomTreeNodeBase<NodeT> *NodeA = getNode(A);
    DomTreeNodeBase<NodeT> *NodeB = getNode(B);

    // Collect NodeA dominators set.
    SmallPtrSet<DomTreeNodeBase<NodeT>*, 16> NodeADoms;
    NodeADoms.insert(NodeA);
    DomTreeNodeBase<NodeT> *IDomA = NodeA->getIDom();
    while (IDomA) {
      NodeADoms.insert(IDomA);
      IDomA = IDomA->getIDom();
    }

    // Walk NodeB immediate dominators chain and find common dominator node.
    DomTreeNodeBase<NodeT> *IDomB = NodeB->getIDom();
    while(IDomB) {
      if (NodeADoms.count(IDomB) != 0)
        return IDomB->getBlock();

      IDomB = IDomB->getIDom();
    }

    return NULL;
  }

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// addNewBlock - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of DomBB dominator node,linking it into 
  /// the children list of the immediate dominator.
  DomTreeNodeBase<NodeT> *addNewBlock(NodeT *BB, NodeT *DomBB) {
    assert(getNode(BB) == 0 && "Block already in dominator tree!");
    DomTreeNodeBase<NodeT> *IDomNode = getNode(DomBB);
    assert(IDomNode && "Not immediate dominator specified for block!");
    DFSInfoValid = false;
    return DomTreeNodes[BB] = 
      IDomNode->addChild(new DomTreeNodeBase<NodeT>(BB, IDomNode));
  }

  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  void changeImmediateDominator(DomTreeNodeBase<NodeT> *N,
                                DomTreeNodeBase<NodeT> *NewIDom) {
    assert(N && NewIDom && "Cannot change null node pointers!");
    DFSInfoValid = false;
    N->setIDom(NewIDom);
  }

  void changeImmediateDominator(NodeT *BB, NodeT *NewBB) {
    changeImmediateDominator(getNode(BB), getNode(NewBB));
  }

  /// eraseNode - Removes a node from  the dominator tree. Block must not
  /// domiante any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  void eraseNode(NodeT *BB) {
    DomTreeNodeBase<NodeT> *Node = getNode(BB);
    assert (Node && "Removing node that isn't in dominator tree.");
    assert (Node->getChildren().empty() && "Node is not a leaf node.");

      // Remove node from immediate dominator's children list.
    DomTreeNodeBase<NodeT> *IDom = Node->getIDom();
    if (IDom) {
      typename std::vector<DomTreeNodeBase<NodeT>*>::iterator I =
        std::find(IDom->Children.begin(), IDom->Children.end(), Node);
      assert(I != IDom->Children.end() &&
             "Not in immediate dominator children set!");
      // I am no longer your child...
      IDom->Children.erase(I);
    }

    DomTreeNodes.erase(BB);
    delete Node;
  }

  /// removeNode - Removes a node from the dominator tree.  Block must not
  /// dominate any other blocks.  Invalidates any node pointing to removed
  /// block.
  void removeNode(NodeT *BB) {
    assert(getNode(BB) && "Removing node that isn't in dominator tree.");
    DomTreeNodes.erase(BB);
  }
  
  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  void splitBlock(NodeT* NewBB) {
    if (this->IsPostDominators)
      this->Split<Inverse<NodeT*>, GraphTraits<Inverse<NodeT*> > >(*this, NewBB);
    else
      this->Split<NodeT*, GraphTraits<NodeT*> >(*this, NewBB);
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &o, const Module* ) const {
    o << "=============================--------------------------------\n";
    if (this->isPostDominator())
      o << "Inorder PostDominator Tree: ";
    else
      o << "Inorder Dominator Tree: ";
    if (this->DFSInfoValid)
      o << "DFSNumbers invalid: " << SlowQueries << " slow queries.";
    o << "\n";

    PrintDomTree<NodeT>(getRootNode(), o, 1);
  }
  
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
  
  virtual void dump() {
    print(llvm::cerr);
  }
  
protected:
  template<class GraphT>
  friend void Compress(DominatorTreeBase<typename GraphT::NodeType>& DT,
                       typename GraphT::NodeType* VIn);

  template<class GraphT>
  friend typename GraphT::NodeType* Eval(
                               DominatorTreeBase<typename GraphT::NodeType>& DT,
                                         typename GraphT::NodeType* V);

  template<class GraphT>
  friend void Link(DominatorTreeBase<typename GraphT::NodeType>& DT,
                   unsigned DFSNumV, typename GraphT::NodeType* W,
         typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &WInfo);
  
  template<class GraphT>
  friend unsigned DFSPass(DominatorTreeBase<typename GraphT::NodeType>& DT,
                          typename GraphT::NodeType* V,
                          unsigned N);
  
  template<class FuncT, class N>
  friend void Calculate(DominatorTreeBase<typename GraphTraits<N>::NodeType>& DT,
                        FuncT& F);
  
  /// updateDFSNumbers - Assign In and Out numbers to the nodes while walking
  /// dominator tree in dfs order.
  void updateDFSNumbers() {
    unsigned DFSNum = 0;

    SmallVector<std::pair<DomTreeNodeBase<NodeT>*,
                typename DomTreeNodeBase<NodeT>::iterator>, 32> WorkStack;

    for (unsigned i = 0, e = (unsigned)this->Roots.size(); i != e; ++i) {
      DomTreeNodeBase<NodeT> *ThisRoot = getNode(this->Roots[i]);
      WorkStack.push_back(std::make_pair(ThisRoot, ThisRoot->begin()));
      ThisRoot->DFSNumIn = DFSNum++;

      while (!WorkStack.empty()) {
        DomTreeNodeBase<NodeT> *Node = WorkStack.back().first;
        typename DomTreeNodeBase<NodeT>::iterator ChildIt =
                                                        WorkStack.back().second;

        // If we visited all of the children of this node, "recurse" back up the
        // stack setting the DFOutNum.
        if (ChildIt == Node->end()) {
          Node->DFSNumOut = DFSNum++;
          WorkStack.pop_back();
        } else {
          // Otherwise, recursively visit this child.
          DomTreeNodeBase<NodeT> *Child = *ChildIt;
          ++WorkStack.back().second;
          
          WorkStack.push_back(std::make_pair(Child, Child->begin()));
          Child->DFSNumIn = DFSNum++;
        }
      }
    }
    
    SlowQueries = 0;
    DFSInfoValid = true;
  }
  
  DomTreeNodeBase<NodeT> *getNodeForBlock(NodeT *BB) {
    if (DomTreeNodeBase<NodeT> *BBNode = this->DomTreeNodes[BB])
      return BBNode;

    // Haven't calculated this node yet?  Get or calculate the node for the
    // immediate dominator.
    NodeT *IDom = getIDom(BB);

    assert(IDom || this->DomTreeNodes[NULL]);
    DomTreeNodeBase<NodeT> *IDomNode = getNodeForBlock(IDom);

    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    DomTreeNodeBase<NodeT> *C = new DomTreeNodeBase<NodeT>(BB, IDomNode);
    return this->DomTreeNodes[BB] = IDomNode->addChild(C);
  }
  
  inline NodeT *getIDom(NodeT *BB) const {
    typename DenseMap<NodeT*, NodeT*>::const_iterator I = IDoms.find(BB);
    return I != IDoms.end() ? I->second : 0;
  }
  
  inline void addRoot(NodeT* BB) {
    this->Roots.push_back(BB);
  }
  
public:
  /// recalculate - compute a dominator tree for the given function
  template<class FT>
  void recalculate(FT& F) {
    if (!this->IsPostDominators) {
      reset();
      
      // Initialize roots
      this->Roots.push_back(&F.front());
      this->IDoms[&F.front()] = 0;
      this->DomTreeNodes[&F.front()] = 0;
      this->Vertex.push_back(0);
      
      Calculate<FT, NodeT*>(*this, F);
      
      updateDFSNumbers();
    } else {
      reset();     // Reset from the last time we were run...

      // Initialize the roots list
      for (typename FT::iterator I = F.begin(), E = F.end(); I != E; ++I) {
        if (std::distance(GraphTraits<FT*>::child_begin(I),
                          GraphTraits<FT*>::child_end(I)) == 0)
          addRoot(I);

        // Prepopulate maps so that we don't get iterator invalidation issues later.
        this->IDoms[I] = 0;
        this->DomTreeNodes[I] = 0;
      }

      this->Vertex.push_back(0);
      
      Calculate<FT, Inverse<NodeT*> >(*this, F);
    }
  }
};

EXTERN_TEMPLATE_INSTANTIATION(class DominatorTreeBase<BasicBlock>);

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class DominatorTree : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTreeBase<BasicBlock>* DT;
  
  DominatorTree() : FunctionPass(&ID) {
    DT = new DominatorTreeBase<BasicBlock>(false);
  }
  
  ~DominatorTree() {
    DT->releaseMemory();
    delete DT;
  }
  
  DominatorTreeBase<BasicBlock>& getBase() { return *DT; }
  
  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BasicBlock*> &getRoots() const {
    return DT->getRoots();
  }
  
  inline BasicBlock *getRoot() const {
    return DT->getRoot();
  }
  
  inline DomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }

  /// compare - Return false if the other dominator tree matches this
  /// dominator tree. Otherwise return true.
  inline bool compare(DominatorTree &Other) const {
    DomTreeNode *R = getRootNode();
    DomTreeNode *OtherR = Other.getRootNode();
    
    if (!R || !OtherR || R->getBlock() != OtherR->getBlock())
      return true;
    
    if (DT->compare(Other.getBase()))
      return true;

    return false;
  }

  virtual bool runOnFunction(Function &F);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  
  inline bool dominates(DomTreeNode* A, DomTreeNode* B) const {
    return DT->dominates(A, B);
  }
  
  inline bool dominates(BasicBlock* A, BasicBlock* B) const {
    return DT->dominates(A, B);
  }
  
  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(Instruction *A, Instruction *B) const {
    BasicBlock *BBA = A->getParent(), *BBB = B->getParent();
    if (BBA != BBB) return DT->dominates(BBA, BBB);

    // It is not possible to determine dominance between two PHI nodes 
    // based on their ordering.
    if (isa<PHINode>(A) && isa<PHINode>(B)) 
      return false;

    // Loop through the basic block until we find A or B.
    BasicBlock::iterator I = BBA->begin();
    for (; &*I != A && &*I != B; ++I) /*empty*/;

    //if(!DT.IsPostDominators) {
      // A dominates B if it is found first in the basic block.
      return &*I == A;
    //} else {
    //  // A post-dominates B if B is found first in the basic block.
    //  return &*I == B;
    //}
  }
  
  inline bool properlyDominates(const DomTreeNode* A, DomTreeNode* B) const {
    return DT->properlyDominates(A, B);
  }
  
  inline bool properlyDominates(BasicBlock* A, BasicBlock* B) const {
    return DT->properlyDominates(A, B);
  }
  
  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  inline BasicBlock *findNearestCommonDominator(BasicBlock *A, BasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }
  
  inline DomTreeNode *operator[](BasicBlock *BB) const {
    return DT->getNode(BB);
  }
  
  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline DomTreeNode *getNode(BasicBlock *BB) const {
    return DT->getNode(BB);
  }
  
  /// addNewBlock - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of DomBB dominator node,linking it into 
  /// the children list of the immediate dominator.
  inline DomTreeNode *addNewBlock(BasicBlock *BB, BasicBlock *DomBB) {
    return DT->addNewBlock(BB, DomBB);
  }
  
  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  inline void changeImmediateDominator(BasicBlock *N, BasicBlock* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }
  
  inline void changeImmediateDominator(DomTreeNode *N, DomTreeNode* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }
  
  /// eraseNode - Removes a node from  the dominator tree. Block must not
  /// domiante any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  inline void eraseNode(BasicBlock *BB) {
    DT->eraseNode(BB);
  }
  
  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  inline void splitBlock(BasicBlock* NewBB) {
    DT->splitBlock(NewBB);
  }
  
  bool isReachableFromEntry(BasicBlock* A) {
    return DT->isReachableFromEntry(A);
  }
  
  
  virtual void releaseMemory() { 
    DT->releaseMemory();
  }
  
  virtual void print(std::ostream &OS, const Module* M= 0) const {
    DT->print(OS, M);
  }
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///
template <> struct GraphTraits<DomTreeNode *> {
  typedef DomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;
  
  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType* N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType* N) {
    return N->end();
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode *> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }
};


//===----------------------------------------------------------------------===//
/// DominanceFrontierBase - Common base class for computing forward and inverse
/// dominance frontiers for a function.
///
class DominanceFrontierBase : public FunctionPass {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
  std::vector<BasicBlock*> Roots;
  const bool IsPostDominators;
  
public:
  DominanceFrontierBase(void *ID, bool isPostDom) 
    : FunctionPass(ID), IsPostDominators(isPostDom) {}

  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BasicBlock*> &getRoots() const { return Roots; }
  
  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::iterator iterator;
  typedef DomSetMapType::const_iterator const_iterator;
  iterator       begin()       { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator       end()         { return Frontiers.end(); }
  const_iterator end()   const { return Frontiers.end(); }
  iterator       find(BasicBlock *B)       { return Frontiers.find(B); }
  const_iterator find(BasicBlock *B) const { return Frontiers.find(B); }

  void addBasicBlock(BasicBlock *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    Frontiers.insert(std::make_pair(BB, frontier));
  }

  /// removeBlock - Remove basic block BB's frontier.
  void removeBlock(BasicBlock *BB) {
    assert(find(BB) != end() && "Block is not in DominanceFrontier!");
    for (iterator I = begin(), E = end(); I != E; ++I)
      I->second.erase(BB);
    Frontiers.erase(BB);
  }

  void addToFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    I->second.insert(Node);
  }

  void removeFromFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    assert(I->second.count(Node) && "Node is not in DominanceFrontier of BB");
    I->second.erase(Node);
  }

  /// compareDomSet - Return false if two domsets match. Otherwise
  /// return true;
  bool compareDomSet(DomSetType &DS1, const DomSetType &DS2) const {
    std::set<BasicBlock *> tmpSet;
    for (DomSetType::const_iterator I = DS2.begin(),
           E = DS2.end(); I != E; ++I) 
      tmpSet.insert(*I);

    for (DomSetType::const_iterator I = DS1.begin(),
           E = DS1.end(); I != E; ++I) {
      BasicBlock *Node = *I;

      if (tmpSet.erase(Node) == 0)
        // Node is in DS1 but not in DS2.
        return true;
    }

    if(!tmpSet.empty())
      // There are nodes that are in DS2 but not in DS1.
      return true;

    // DS1 and DS2 matches.
    return false;
  }

  /// compare - Return true if the other dominance frontier base matches
  /// this dominance frontier base. Otherwise return false.
  bool compare(DominanceFrontierBase &Other) const {
    DomSetMapType tmpFrontiers;
    for (DomSetMapType::const_iterator I = Other.begin(),
           E = Other.end(); I != E; ++I) 
      tmpFrontiers.insert(std::make_pair(I->first, I->second));

    for (DomSetMapType::iterator I = tmpFrontiers.begin(),
           E = tmpFrontiers.end(); I != E; ++I) {
      BasicBlock *Node = I->first;
      const_iterator DFI = find(Node);
      if (DFI == end()) 
        return true;

      if (compareDomSet(I->second, DFI->second))
        return true;

      tmpFrontiers.erase(Node);
    }

    if (!tmpFrontiers.empty())
      return true;

    return false;
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
  virtual void dump();
};


//===-------------------------------------
/// DominanceFrontier Class - Concrete subclass of DominanceFrontierBase that is
/// used to compute a forward dominator frontiers.
///
class DominanceFrontier : public DominanceFrontierBase {
public:
  static char ID; // Pass ID, replacement for typeid
  DominanceFrontier() : 
    DominanceFrontierBase(&ID, false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    assert(Roots.size() == 1 && "Only one entry block for forward domfronts!");
    calculate(DT, DT[Roots[0]]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }

  /// splitBlock - BB is split and now it has one successor. Update dominance
  /// frontier to reflect this change.
  void splitBlock(BasicBlock *BB);

  /// BasicBlock BB's new dominator is NewBB. Update BB's dominance frontier
  /// to reflect this change.
  void changeImmediateDominator(BasicBlock *BB, BasicBlock *NewBB,
                                DominatorTree *DT) {
    // NewBB is now  dominating BB. Which means BB's dominance
    // frontier is now part of NewBB's dominance frontier. However, BB
    // itself is not member of NewBB's dominance frontier.
    DominanceFrontier::iterator NewDFI = find(NewBB);
    DominanceFrontier::iterator DFI = find(BB);
    // If BB was an entry block then its frontier is empty.
    if (DFI == end())
      return;
    DominanceFrontier::DomSetType BBSet = DFI->second;
    for (DominanceFrontier::DomSetType::iterator BBSetI = BBSet.begin(),
           BBSetE = BBSet.end(); BBSetI != BBSetE; ++BBSetI) {
      BasicBlock *DFMember = *BBSetI;
      // Insert only if NewBB dominates DFMember.
      if (!DT->dominates(NewBB, DFMember))
        NewDFI->second.insert(DFMember);
    }
    NewDFI->second.erase(BB);
  }

  const DomSetType &calculate(const DominatorTree &DT,
                              const DomTreeNode *Node);
};


} // End llvm namespace

#endif
