//===- llvm/Analysis/Dominators.h - Dominator Info Calculation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DominatorTree class, which provides fast and efficient
// dominance queries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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

  /// getRoots - Return the root blocks of the current CFG.  This may include
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
    for (iterator I = Other->begin(), E = Other->end(); I != E; ++I) {
      NodeT *Nd = (*I)->getBlock();
      OtherChildren.insert(Nd);
    }

    for (iterator I = begin(), E = end(); I != E; ++I) {
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
inline raw_ostream &operator<<(raw_ostream &o,
                               const DomTreeNodeBase<NodeT> *Node) {
  if (Node->getBlock())
    WriteAsOperand(o, Node->getBlock(), false);
  else
    o << " <<exit node>>";

  o << " {" << Node->getDFSNumIn() << "," << Node->getDFSNumOut() << "}";

  return o << "\n";
}

template<class NodeT>
inline void PrintDomTree(const DomTreeNodeBase<NodeT> *N, raw_ostream &o,
                         unsigned Lev) {
  o.indent(2*Lev) << "[" << Lev << "] " << N;
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
  bool dominatedBySlowTreeWalk(const DomTreeNodeBase<NodeT> *A,
                               const DomTreeNodeBase<NodeT> *B) const {
    assert(A != B);
    assert(isReachableFromEntry(B));
    assert(isReachableFromEntry(A));

    const DomTreeNodeBase<NodeT> *IDom;
    while ((IDom = B->getIDom()) != 0 && IDom != A && IDom != B)
      B = IDom;   // Walk up the tree
    return IDom != 0;
  }

protected:
  typedef DenseMap<NodeT*, DomTreeNodeBase<NodeT>*> DomTreeNodeMapType;
  DomTreeNodeMapType DomTreeNodes;
  DomTreeNodeBase<NodeT> *RootNode;

  bool DFSInfoValid;
  unsigned int SlowQueries;
  // Information record used during immediate dominators computation.
  struct InfoRec {
    unsigned DFSNum;
    unsigned Parent;
    unsigned Semi;
    NodeT *Label;

    InfoRec() : DFSNum(0), Parent(0), Semi(0), Label(0) {}
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
    assert(std::distance(GraphT::child_begin(NewBB),
                         GraphT::child_end(NewBB)) == 1 &&
           "NewBB should have a single successor!");
    typename GraphT::NodeType* NewBBSucc = *GraphT::child_begin(NewBB);

    std::vector<typename GraphT::NodeType*> PredBlocks;
    typedef GraphTraits<Inverse<N> > InvTraits;
    for (typename InvTraits::ChildIteratorType PI =
         InvTraits::child_begin(NewBB),
         PE = InvTraits::child_end(NewBB); PI != PE; ++PI)
      PredBlocks.push_back(*PI);

    assert(!PredBlocks.empty() && "No predblocks?");

    bool NewBBDominatesNewBBSucc = true;
    for (typename InvTraits::ChildIteratorType PI =
         InvTraits::child_begin(NewBBSucc),
         E = InvTraits::child_end(NewBBSucc); PI != E; ++PI) {
      typename InvTraits::NodeType *ND = *PI;
      if (ND != NewBB && !DT.dominates(NewBBSucc, ND) &&
          DT.isReachableFromEntry(ND)) {
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

    // It's possible that none of the predecessors of NewBB are reachable;
    // in that case, NewBB itself is unreachable, so nothing needs to be
    // changed.
    if (!NewBBIDom)
      return;

    for (i = i + 1; i < PredBlocks.size(); ++i) {
      if (DT.isReachableFromEntry(PredBlocks[i]))
        NewBBIDom = DT.findNearestCommonDominator(NewBBIDom, PredBlocks[i]);
    }

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

  /// compare - Return false if the other dominator tree base matches this
  /// dominator tree base. Otherwise return true.
  bool compare(DominatorTreeBase &Other) const {

    const DomTreeNodeMapType &OtherDomTreeNodes = Other.DomTreeNodes;
    if (DomTreeNodes.size() != OtherDomTreeNodes.size())
      return true;

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
    return DomTreeNodes.lookup(BB);
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

  /// properlyDominates - Returns true iff A dominates B and A != B.
  /// Note that this is not a constant time operation!
  ///
  bool properlyDominates(const DomTreeNodeBase<NodeT> *A,
                         const DomTreeNodeBase<NodeT> *B) {
    if (A == 0 || B == 0)
      return false;
    if (A == B)
      return false;
    return dominates(A, B);
  }

  bool properlyDominates(const NodeT *A, const NodeT *B);

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  bool isReachableFromEntry(const NodeT* A) const {
    assert(!this->isPostDominator() &&
           "This is not implemented for post dominators");
    return isReachableFromEntry(getNode(const_cast<NodeT *>(A)));
  }

  inline bool isReachableFromEntry(const DomTreeNodeBase<NodeT> *A) const {
    return A;
  }

  /// dominates - Returns true iff A dominates B.  Note that this is not a
  /// constant time operation!
  ///
  inline bool dominates(const DomTreeNodeBase<NodeT> *A,
                        const DomTreeNodeBase<NodeT> *B) {
    // A node trivially dominates itself.
    if (B == A)
      return true;

    // An unreachable node is dominated by anything.
    if (!isReachableFromEntry(B))
      return true;

    // And dominates nothing.
    if (!isReachableFromEntry(A))
      return false;

    // Compare the result of the tree walk and the dfs numbers, if expensive
    // checks are enabled.
#ifdef XDEBUG
    assert((!DFSInfoValid ||
            (dominatedBySlowTreeWalk(A, B) == B->DominatedBy(A))) &&
           "Tree walk disagrees with dfs numbers!");
#endif

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

  bool dominates(const NodeT *A, const NodeT *B);

  NodeT *getRoot() const {
    assert(this->Roots.size() == 1 && "Should always have entry node!");
    return this->Roots[0];
  }

  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  NodeT *findNearestCommonDominator(NodeT *A, NodeT *B) {
    assert(A->getParent() == B->getParent() &&
           "Two blocks are not in same function");

    // If either A or B is a entry block then it is nearest common dominator
    // (for forward-dominators).
    if (!this->isPostDominator()) {
      NodeT &Entry = A->getParent()->front();
      if (A == &Entry || B == &Entry)
        return &Entry;
    }

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
    while (IDomB) {
      if (NodeADoms.count(IDomB) != 0)
        return IDomB->getBlock();

      IDomB = IDomB->getIDom();
    }

    return NULL;
  }

  const NodeT *findNearestCommonDominator(const NodeT *A, const NodeT *B) {
    // Cast away the const qualifiers here. This is ok since
    // const is re-introduced on the return type.
    return findNearestCommonDominator(const_cast<NodeT *>(A),
                                      const_cast<NodeT *>(B));
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

  /// eraseNode - Removes a node from the dominator tree. Block must not
  /// dominate any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  void eraseNode(NodeT *BB) {
    DomTreeNodeBase<NodeT> *Node = getNode(BB);
    assert(Node && "Removing node that isn't in dominator tree.");
    assert(Node->getChildren().empty() && "Node is not a leaf node.");

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
  void print(raw_ostream &o) const {
    o << "=============================--------------------------------\n";
    if (this->isPostDominator())
      o << "Inorder PostDominator Tree: ";
    else
      o << "Inorder Dominator Tree: ";
    if (!this->DFSInfoValid)
      o << "DFSNumbers invalid: " << SlowQueries << " slow queries.";
    o << "\n";

    // The postdom tree can have a null root if there are no returns.
    if (getRootNode())
      PrintDomTree<NodeT>(getRootNode(), o, 1);
  }

protected:
  template<class GraphT>
  friend typename GraphT::NodeType* Eval(
                               DominatorTreeBase<typename GraphT::NodeType>& DT,
                                         typename GraphT::NodeType* V,
                                         unsigned LastLinked);

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

    DomTreeNodeBase<NodeT> *ThisRoot = getRootNode();

    if (!ThisRoot)
      return;

    // Even in the case of multiple exits that form the post dominator root
    // nodes, do not iterate over all exits, but start from the virtual root
    // node. Otherwise bbs, that are not post dominated by any exit but by the
    // virtual root node, will never be assigned a DFS number.
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

    SlowQueries = 0;
    DFSInfoValid = true;
  }

  DomTreeNodeBase<NodeT> *getNodeForBlock(NodeT *BB) {
    if (DomTreeNodeBase<NodeT> *Node = getNode(BB))
      return Node;

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
    return IDoms.lookup(BB);
  }

  inline void addRoot(NodeT* BB) {
    this->Roots.push_back(BB);
  }

public:
  /// recalculate - compute a dominator tree for the given function
  template<class FT>
  void recalculate(FT& F) {
    typedef GraphTraits<FT*> TraitsTy;
    reset();
    this->Vertex.push_back(0);

    if (!this->IsPostDominators) {
      // Initialize root
      NodeT *entry = TraitsTy::getEntryNode(&F);
      this->Roots.push_back(entry);
      this->IDoms[entry] = 0;
      this->DomTreeNodes[entry] = 0;

      Calculate<FT, NodeT*>(*this, F);
    } else {
      // Initialize the roots list
      for (typename TraitsTy::nodes_iterator I = TraitsTy::nodes_begin(&F),
                                        E = TraitsTy::nodes_end(&F); I != E; ++I) {
        if (std::distance(TraitsTy::child_begin(I),
                          TraitsTy::child_end(I)) == 0)
          addRoot(I);

        // Prepopulate maps so that we don't get iterator invalidation issues later.
        this->IDoms[I] = 0;
        this->DomTreeNodes[I] = 0;
      }

      Calculate<FT, Inverse<NodeT*> >(*this, F);
    }
  }
};

// These two functions are declared out of line as a workaround for building
// with old (< r147295) versions of clang because of pr11642.
template<class NodeT>
bool DominatorTreeBase<NodeT>::dominates(const NodeT *A, const NodeT *B) {
  if (A == B)
    return true;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(getNode(const_cast<NodeT *>(A)),
                   getNode(const_cast<NodeT *>(B)));
}
template<class NodeT>
bool
DominatorTreeBase<NodeT>::properlyDominates(const NodeT *A, const NodeT *B) {
  if (A == B)
    return false;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(getNode(const_cast<NodeT *>(A)),
                   getNode(const_cast<NodeT *>(B)));
}

EXTERN_TEMPLATE_INSTANTIATION(class DominatorTreeBase<BasicBlock>);

class BasicBlockEdge {
  const BasicBlock *Start;
  const BasicBlock *End;
public:
  BasicBlockEdge(const BasicBlock *Start_, const BasicBlock *End_) :
    Start(Start_), End(End_) { }
  const BasicBlock *getStart() const {
    return Start;
  }
  const BasicBlock *getEnd() const {
    return End;
  }
  bool isSingleEdge() const;
};

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class DominatorTree : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTreeBase<BasicBlock>* DT;

  DominatorTree() : FunctionPass(ID) {
    initializeDominatorTreePass(*PassRegistry::getPassRegistry());
    DT = new DominatorTreeBase<BasicBlock>(false);
  }

  ~DominatorTree() {
    delete DT;
  }

  DominatorTreeBase<BasicBlock>& getBase() { return *DT; }

  /// getRoots - Return the root blocks of the current CFG.  This may include
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

  virtual void verifyAnalysis() const;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  inline bool dominates(const DomTreeNode* A, const DomTreeNode* B) const {
    return DT->dominates(A, B);
  }

  inline bool dominates(const BasicBlock* A, const BasicBlock* B) const {
    return DT->dominates(A, B);
  }

  // dominates - Return true if Def dominates a use in User. This performs
  // the special checks necessary if Def and User are in the same basic block.
  // Note that Def doesn't dominate a use in Def itself!
  bool dominates(const Instruction *Def, const Use &U) const;
  bool dominates(const Instruction *Def, const Instruction *User) const;
  bool dominates(const Instruction *Def, const BasicBlock *BB) const;
  bool dominates(const BasicBlockEdge &BBE, const Use &U) const;
  bool dominates(const BasicBlockEdge &BBE, const BasicBlock *BB) const;

  bool properlyDominates(const DomTreeNode *A, const DomTreeNode *B) const {
    return DT->properlyDominates(A, B);
  }

  bool properlyDominates(const BasicBlock *A, const BasicBlock *B) const {
    return DT->properlyDominates(A, B);
  }

  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  inline BasicBlock *findNearestCommonDominator(BasicBlock *A, BasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  inline const BasicBlock *findNearestCommonDominator(const BasicBlock *A,
                                                      const BasicBlock *B) {
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

  /// eraseNode - Removes a node from the dominator tree. Block must not
  /// dominate any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  inline void eraseNode(BasicBlock *BB) {
    DT->eraseNode(BB);
  }

  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  inline void splitBlock(BasicBlock* NewBB) {
    DT->splitBlock(NewBB);
  }

  bool isReachableFromEntry(const BasicBlock* A) const {
    return DT->isReachableFromEntry(A);
  }

  bool isReachableFromEntry(const Use &U) const;


  virtual void releaseMemory() {
    DT->releaseMemory();
  }

  virtual void print(raw_ostream &OS, const Module* M= 0) const;
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///
template <> struct GraphTraits<DomTreeNode*> {
  typedef DomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }

  typedef df_iterator<DomTreeNode*> nodes_iterator;

  static nodes_iterator nodes_begin(DomTreeNode *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DomTreeNode *N) {
    return df_end(getEntryNode(N));
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(DominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};


} // End llvm namespace

#endif
