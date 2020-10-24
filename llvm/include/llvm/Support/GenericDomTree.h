//===- GenericDomTree.h - Generic dominator trees for graphs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a set of templates that efficiently compute a dominator
/// tree over a generic graph. This is used typically in LLVM for fast
/// dominance queries on the CFG, but is fully generic w.r.t. the underlying
/// graph types.
///
/// Unlike ADT/* graph algorithms, generic dominator tree has more requirements
/// on the graph's NodeRef:
///  * The NodeRef should be a pointer.
///  * NodeRef->getParent() must return the parent node that is also a pointer.
///  * CfgTraitsFor<NodeType> must be implemented, though a partial
///    implementation without the "value" parts of CfgTraits is sufficient.
///
/// FIXME: Should GenericDomTree be implemented entirely in terms of CfgTraits?
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICDOMTREE_H
#define LLVM_SUPPORT_GENERICDOMTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CFGDiff.h"
#include "llvm/Support/CFGUpdate.h"
#include "llvm/Support/CfgTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace llvm {

class GenericDominatorTreeBase;

template <typename NodeT, bool IsPostDom>
class DominatorTreeBase;

namespace DomTreeBuilder {
template <typename DomTreeT>
struct SemiNCAInfo;
}  // namespace DomTreeBuilder

/// Type-erased base class for dominator tree nodes. Can be used for generic
/// read-only queries on a dominator tree.
class GenericDomTreeNodeBase {
  friend GenericDominatorTreeBase;
  template <typename NodeT, bool IsPostDom> friend class DominatorTreeBase;
  template <typename DomTreeT> friend struct DomTreeBuilder::SemiNCAInfo;

protected:
  CfgBlockRef TheBB;
  GenericDomTreeNodeBase *IDom;
  unsigned Level;
  SmallVector<GenericDomTreeNodeBase *, 4> Children;
  mutable unsigned DFSNumIn = ~0;
  mutable unsigned DFSNumOut = ~0;

public:
  GenericDomTreeNodeBase(CfgBlockRef BB, GenericDomTreeNodeBase *idom)
      : TheBB(BB), IDom(idom), Level(idom ? idom->Level + 1 : 0) {}

  using iterator = typename SmallVector<GenericDomTreeNodeBase *, 4>::iterator;
  using const_iterator =
      typename SmallVector<GenericDomTreeNodeBase *, 4>::const_iterator;

  iterator begin() { return Children.begin(); }
  iterator end() { return Children.end(); }
  const_iterator begin() const { return Children.begin(); }
  const_iterator end() const { return Children.end(); }

  GenericDomTreeNodeBase *const &back() const { return Children.back(); }

  iterator_range<iterator> children() { return make_range(begin(), end()); }
  iterator_range<const_iterator> children() const {
    return make_range(begin(), end());
  }

  CfgBlockRef getBlock() const { return TheBB; }
  GenericDomTreeNodeBase *getIDom() const { return IDom; }
  unsigned getLevel() const { return Level; }

  bool isLeaf() const { return Children.empty(); }
  size_t getNumChildren() const { return Children.size(); }

  void clearAllChildren() { Children.clear(); }

  bool compare(const GenericDomTreeNodeBase *Other) const;
  void setIDom(GenericDomTreeNodeBase *NewIDom);

  /// getDFSNumIn/getDFSNumOut - These return the DFS visitation order for nodes
  /// in the dominator tree. They are only guaranteed valid if
  /// updateDFSNumbers() has been called.
  unsigned getDFSNumIn() const { return DFSNumIn; }
  unsigned getDFSNumOut() const { return DFSNumOut; }

  std::unique_ptr<GenericDomTreeNodeBase>
  addChild(std::unique_ptr<GenericDomTreeNodeBase> C) {
    Children.push_back(C.get());
    return C;
  }

private:
  // Return true if this node is dominated by other. Use this only if DFS info
  // is valid.
  bool DominatedBy(const GenericDomTreeNodeBase *other) const {
    return this->DFSNumIn >= other->DFSNumIn &&
           this->DFSNumOut <= other->DFSNumOut;
  }

  void UpdateLevel();
};

/// Base class for the actual dominator tree node.
template <class NodeT> class DomTreeNodeBase : public GenericDomTreeNodeBase {
  using CfgTraits = typename CfgTraitsFor<NodeT>::CfgTraits;

  friend class PostDominatorTree;
  friend class DominatorTreeBase<NodeT, false>;
  friend class DominatorTreeBase<NodeT, true>;
  friend struct DomTreeBuilder::SemiNCAInfo<DominatorTreeBase<NodeT, false>>;
  friend struct DomTreeBuilder::SemiNCAInfo<DominatorTreeBase<NodeT, true>>;

public:
  DomTreeNodeBase(NodeT *BB, DomTreeNodeBase *IDom)
      : GenericDomTreeNodeBase(CfgTraits::wrapRef(BB), IDom) {}

  struct const_iterator;

  using const_iterator_base = iterator_adaptor_base<
      const_iterator, GenericDomTreeNodeBase::const_iterator,
      typename std::iterator_traits<
          GenericDomTreeNodeBase::const_iterator>::iterator_category,
      // value_type
      DomTreeNodeBase *,
      typename std::iterator_traits<
          GenericDomTreeNodeBase::const_iterator>::difference_type,
      // pointer (not really usable, but we need to put something here)
      DomTreeNodeBase *const *,
      // reference (not a true reference, because operator* doesn't return one)
      DomTreeNodeBase *>;

  struct const_iterator : const_iterator_base {
    const_iterator() = default;
    explicit const_iterator(GenericDomTreeNodeBase::const_iterator it)
        : const_iterator_base(it) {}

    auto operator*() const { return static_cast<DomTreeNodeBase *>(*this->I); }
  };

  auto begin() const { return const_iterator{GenericDomTreeNodeBase::begin()}; }
  auto end() const { return const_iterator{GenericDomTreeNodeBase::end()}; }

  DomTreeNodeBase *back() const {
    return static_cast<DomTreeNodeBase *>(Children.back());
  }

  iterator_range<const_iterator> children() const {
    return make_range(begin(), end());
  }

  NodeT *getBlock() const { return CfgTraits::unwrapRef(TheBB); }
  DomTreeNodeBase *getIDom() const {
    return static_cast<DomTreeNodeBase *>(IDom);
  }
};

template <class NodeT>
raw_ostream &operator<<(raw_ostream &O, const DomTreeNodeBase<NodeT> *Node) {
  if (Node->getBlock())
    Node->getBlock()->printAsOperand(O, false);
  else
    O << " <<exit node>>";

  O << " {" << Node->getDFSNumIn() << "," << Node->getDFSNumOut() << "} ["
    << Node->getLevel() << "]\n";

  return O;
}

template <class NodeT>
void PrintDomTree(const DomTreeNodeBase<NodeT> *N, raw_ostream &O,
                  unsigned Lev) {
  O.indent(2 * Lev) << "[" << Lev << "] " << N;
  for (const DomTreeNodeBase<NodeT> *Child : N->children())
    PrintDomTree<NodeT>(Child, O, Lev + 1);
}

namespace DomTreeBuilder {
// The routines below are provided in a separate header but referenced here.
template <typename DomTreeT>
void Calculate(DomTreeT &DT);

template <typename DomTreeT>
void CalculateWithUpdates(DomTreeT &DT,
                          ArrayRef<typename DomTreeT::UpdateType> Updates);

template <typename DomTreeT>
void InsertEdge(DomTreeT &DT, typename DomTreeT::NodePtr From,
                typename DomTreeT::NodePtr To);

template <typename DomTreeT>
void DeleteEdge(DomTreeT &DT, typename DomTreeT::NodePtr From,
                typename DomTreeT::NodePtr To);

template <typename DomTreeT>
void ApplyUpdates(DomTreeT &DT,
                  GraphDiff<typename DomTreeT::NodePtr,
                            DomTreeT::IsPostDominator> &PreViewCFG,
                  GraphDiff<typename DomTreeT::NodePtr,
                            DomTreeT::IsPostDominator> *PostViewCFG);

template <typename DomTreeT>
bool Verify(const DomTreeT &DT, typename DomTreeT::VerificationLevel VL);
}  // namespace DomTreeBuilder

/// Type-erased dominator tree base class.
///
/// This base class of all dominator trees can be used for read-only queries
/// on a dominator tree.
class GenericDominatorTreeBase {
protected:
  DenseMap<CfgBlockRef, std::unique_ptr<GenericDomTreeNodeBase>> DomTreeNodes;
  GenericDomTreeNodeBase *RootNode = nullptr;

  mutable bool DFSInfoValid = false;
  mutable unsigned int SlowQueries = 0;

  // Disallow copying
  GenericDominatorTreeBase(const GenericDominatorTreeBase &) = delete;
  GenericDominatorTreeBase &
  operator=(const GenericDominatorTreeBase &) = delete;

public:
  GenericDominatorTreeBase() {}

  GenericDominatorTreeBase(GenericDominatorTreeBase &&Arg)
      : DomTreeNodes(std::move(Arg.DomTreeNodes)), RootNode(Arg.RootNode),
        DFSInfoValid(Arg.DFSInfoValid), SlowQueries(Arg.SlowQueries) {
    Arg.wipe();
  }

  GenericDominatorTreeBase &operator=(GenericDominatorTreeBase &&RHS) {
    DomTreeNodes = std::move(RHS.DomTreeNodes);
    RootNode = RHS.RootNode;
    DFSInfoValid = RHS.DFSInfoValid;
    SlowQueries = RHS.SlowQueries;
    RHS.wipe();
    return *this;
  }

  void reset();

  bool compare(const GenericDominatorTreeBase &Other) const;

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.  The result
  /// may (but is not required to) be null for a forward (backwards)
  /// statically unreachable block.
  GenericDomTreeNodeBase *getNode(CfgBlockRef BB) const {
    auto I = DomTreeNodes.find(BB);
    if (I != DomTreeNodes.end())
      return I->second.get();
    return nullptr;
  }

  /// See getNode.
  GenericDomTreeNodeBase *operator[](CfgBlockRef BB) const {
    return getNode(BB);
  }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  GenericDomTreeNodeBase *getRootNode() { return RootNode; }
  const GenericDomTreeNodeBase *getRootNode() const { return RootNode; }

  bool isReachableFromEntry(const GenericDomTreeNodeBase *A) const { return A; }

  bool properlyDominates(const GenericDomTreeNodeBase *A,
                         const GenericDomTreeNodeBase *B) const;
  bool properlyDominatesBlock(CfgBlockRef A, CfgBlockRef B) const;

  bool dominates(const GenericDomTreeNodeBase *A,
                 const GenericDomTreeNodeBase *B) const;
  bool dominatesBlock(CfgBlockRef A, CfgBlockRef B) const;

  const GenericDomTreeNodeBase *
  findNearestCommonDominator(const GenericDomTreeNodeBase *A,
                             const GenericDomTreeNodeBase *B) const;
  CfgBlockRef findNearestCommonDominatorBlock(CfgBlockRef A,
                                              CfgBlockRef B) const;

  void updateDFSNumbers() const;

private:
  /// Wipe this tree's state without releasing any resources.
  ///
  /// This is essentially a post-move helper only. It leaves the object in an
  /// assignable and destroyable state, but otherwise invalid.
  void wipe() {
    DomTreeNodes.clear();
    RootNode = nullptr;
  }

  bool dominatedBySlowTreeWalk(const GenericDomTreeNodeBase *A,
                               const GenericDomTreeNodeBase *B) const;
};

/// Core dominator tree base class.
///
/// This class is a generic template over graph nodes. It is instantiated for
/// various graphs in the LLVM IR or in the code generator.
template <typename NodeT, bool IsPostDom>
class DominatorTreeBase : public GenericDominatorTreeBase {
public:
  using CfgTraits = typename CfgTraitsFor<NodeT>::CfgTraits;

  static_assert(std::is_pointer<typename GraphTraits<NodeT *>::NodeRef>::value,
                "Currently DominatorTreeBase supports only pointer nodes");
  using NodeType = NodeT;
  using NodePtr = NodeT *;
  using ParentPtr = decltype(std::declval<NodeT *>()->getParent());
  static_assert(std::is_pointer<ParentPtr>::value,
                "Currently NodeT's parent must be a pointer type");
  using ParentType = std::remove_pointer_t<ParentPtr>;
  static constexpr bool IsPostDominator = IsPostDom;

  using UpdateType = cfg::Update<NodePtr>;
  using UpdateKind = cfg::UpdateKind;
  static constexpr UpdateKind Insert = UpdateKind::Insert;
  static constexpr UpdateKind Delete = UpdateKind::Delete;

  enum class VerificationLevel { Fast, Basic, Full };

protected:
  // Dominators always have a single root, postdominators can have more.
  SmallVector<NodeT *, IsPostDom ? 4 : 1> Roots;
  ParentPtr Parent = nullptr;

  friend struct DomTreeBuilder::SemiNCAInfo<DominatorTreeBase>;

public:
  DominatorTreeBase() {}

  /// Iteration over roots.
  ///
  /// This may include multiple blocks if we are computing post dominators.
  /// For forward dominators, this will always be a single block (the entry
  /// block).
  using root_iterator = typename SmallVectorImpl<NodeT *>::iterator;
  using const_root_iterator = typename SmallVectorImpl<NodeT *>::const_iterator;

  root_iterator root_begin() { return Roots.begin(); }
  const_root_iterator root_begin() const { return Roots.begin(); }
  root_iterator root_end() { return Roots.end(); }
  const_root_iterator root_end() const { return Roots.end(); }

  size_t root_size() const { return Roots.size(); }

  iterator_range<root_iterator> roots() {
    return make_range(root_begin(), root_end());
  }
  iterator_range<const_root_iterator> roots() const {
    return make_range(root_begin(), root_end());
  }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominator; }

  /// compare - Return false if the other dominator tree base matches this
  /// dominator tree base. Otherwise return true.
  bool compare(const DominatorTreeBase &Other) const {
    if (Parent != Other.Parent) return true;

    if (Roots.size() != Other.Roots.size())
      return true;

    if (!std::is_permutation(Roots.begin(), Roots.end(), Other.Roots.begin()))
      return true;

    return GenericDominatorTreeBase::compare(Other);
  }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.  The result
  /// may (but is not required to) be null for a forward (backwards)
  /// statically unreachable block.
  DomTreeNodeBase<NodeT> *getNode(const NodeT *BB) const {
    return static_cast<DomTreeNodeBase<NodeT> *>(
        GenericDominatorTreeBase::getNode(
            CfgTraits::wrapRef(const_cast<NodeT *>(BB))));
  }

  /// See getNode.
  DomTreeNodeBase<NodeT> *operator[](const NodeT *BB) const {
    return getNode(BB);
  }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  ///
  DomTreeNodeBase<NodeT> *getRootNode() {
    return static_cast<DomTreeNodeBase<NodeT> *>(RootNode);
  }
  const DomTreeNodeBase<NodeT> *getRootNode() const {
    return static_cast<const DomTreeNodeBase<NodeT> *>(RootNode);
  }

  /// Get all nodes dominated by R, including R itself.
  void getDescendants(NodeT *R, SmallVectorImpl<NodeT *> &Result) const {
    Result.clear();
    const DomTreeNodeBase<NodeT> *RN = getNode(R);
    if (!RN)
      return; // If R is unreachable, it will not be present in the DOM tree.
    SmallVector<const DomTreeNodeBase<NodeT> *, 8> WL;
    WL.push_back(RN);

    while (!WL.empty()) {
      const DomTreeNodeBase<NodeT> *N = WL.pop_back_val();
      Result.push_back(N->getBlock());
      WL.append(N->begin(), N->end());
    }
  }

  bool properlyDominates(const DomTreeNodeBase<NodeT> *A,
                         const DomTreeNodeBase<NodeT> *B) const {
    return GenericDominatorTreeBase::properlyDominates(A, B);
  }
  bool properlyDominates(const NodeT *A, const NodeT *B) const {
    if (A == B)
      return false;
    return GenericDominatorTreeBase::dominates(getNode(A), getNode(B));
  }

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  bool isReachableFromEntry(const NodeT *A) const {
    assert(!this->isPostDominator() &&
           "This is not implemented for post dominators");
    return getNode(const_cast<NodeT *>(A)) != nullptr;
  }
  bool isReachableFromEntry(const DomTreeNodeBase<NodeT> *A) const {
    return A != nullptr;
  }

  bool dominates(const DomTreeNodeBase<NodeT> *A,
                 const DomTreeNodeBase<NodeT> *B) const {
    return GenericDominatorTreeBase::dominates(A, B);
  }
  bool dominates(const NodeT *A, const NodeT *B) const {
    if (A == B)
      return true;
    return GenericDominatorTreeBase::dominates(getNode(A), getNode(B));
  }

  NodeT *getRoot() const {
    assert(this->Roots.size() == 1 && "Should always have entry node!");
    return this->Roots[0];
  }

  bool isVirtualRoot(const DomTreeNodeBase<NodeT> *A) const {
    return isPostDominator() && !A->getBlock();
  }

  const DomTreeNodeBase<NodeT> *
  findNearestCommonDominator(const DomTreeNodeBase<NodeT> *A,
                             const DomTreeNodeBase<NodeT> *B) const {
    return static_cast<const DomTreeNodeBase<NodeT> *>(
        GenericDominatorTreeBase::findNearestCommonDominator(A, B));
  }
  const NodeT *findNearestCommonDominator(const NodeT *A,
                                          const NodeT *B) const {
    assert(A && B && "Pointers are not valid");
    const DomTreeNodeBase<NodeT> *dom =
        static_cast<const DomTreeNodeBase<NodeT> *>(
            GenericDominatorTreeBase::findNearestCommonDominator(getNode(A),
                                                                 getNode(B)));
    return dom->getBlock();
  }
  NodeT *findNearestCommonDominator(NodeT *A, NodeT *B) const {
    assert(A && B && "Pointers are not valid");
    const DomTreeNodeBase<NodeT> *dom =
        static_cast<const DomTreeNodeBase<NodeT> *>(
            GenericDominatorTreeBase::findNearestCommonDominator(getNode(A),
                                                                 getNode(B)));
    return dom->getBlock();
  }

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// Inform the dominator tree about a sequence of CFG edge insertions and
  /// deletions and perform a batch update on the tree.
  ///
  /// This function should be used when there were multiple CFG updates after
  /// the last dominator tree update. It takes care of performing the updates
  /// in sync with the CFG and optimizes away the redundant operations that
  /// cancel each other.
  /// The functions expects the sequence of updates to be balanced. Eg.:
  ///  - {{Insert, A, B}, {Delete, A, B}, {Insert, A, B}} is fine, because
  ///    logically it results in a single insertions.
  ///  - {{Insert, A, B}, {Insert, A, B}} is invalid, because it doesn't make
  ///    sense to insert the same edge twice.
  ///
  /// What's more, the functions assumes that it's safe to ask every node in the
  /// CFG about its children and inverse children. This implies that deletions
  /// of CFG edges must not delete the CFG nodes before calling this function.
  ///
  /// The applyUpdates function can reorder the updates and remove redundant
  /// ones internally. The batch updater is also able to detect sequences of
  /// zero and exactly one update -- it's optimized to do less work in these
  /// cases.
  ///
  /// Note that for postdominators it automatically takes care of applying
  /// updates on reverse edges internally (so there's no need to swap the
  /// From and To pointers when constructing DominatorTree::UpdateType).
  /// The type of updates is the same for DomTreeBase<T> and PostDomTreeBase<T>
  /// with the same template parameter T.
  ///
  /// \param Updates An unordered sequence of updates to perform. The current
  /// CFG and the reverse of these updates provides the pre-view of the CFG.
  ///
  void applyUpdates(ArrayRef<UpdateType> Updates) {
    GraphDiff<NodePtr, IsPostDominator> PreViewCFG(
        Updates, /*ReverseApplyUpdates=*/true);
    DomTreeBuilder::ApplyUpdates(*this, PreViewCFG, nullptr);
  }

  /// \param Updates An unordered sequence of updates to perform. The current
  /// CFG and the reverse of these updates provides the pre-view of the CFG.
  /// \param PostViewUpdates An unordered sequence of update to perform in order
  /// to obtain a post-view of the CFG. The DT will be updates assuming the
  /// obtained PostViewCFG is the desired end state.
  void applyUpdates(ArrayRef<UpdateType> Updates,
                    ArrayRef<UpdateType> PostViewUpdates) {
    // GraphDiff<NodePtr, IsPostDom> *PostViewCFG = nullptr) {
    if (Updates.empty()) {
      GraphDiff<NodePtr, IsPostDom> PostViewCFG(PostViewUpdates);
      DomTreeBuilder::ApplyUpdates(*this, PostViewCFG, &PostViewCFG);
    } else {
      // TODO:
      // PreViewCFG needs to merge Updates and PostViewCFG. The updates in
      // Updates need to be reversed, and match the direction in PostViewCFG.
      // Normally, a PostViewCFG is created without reversing updates, so one
      // of the internal vectors needs reversing in order to do the
      // legalization of the merged vector of updates.
      llvm_unreachable("Currently unsupported to update given a set of "
                       "updates towards a PostView");
    }
  }

  /// Inform the dominator tree about a CFG edge insertion and update the tree.
  ///
  /// This function has to be called just before or just after making the update
  /// on the actual CFG. There cannot be any other updates that the dominator
  /// tree doesn't know about.
  ///
  /// Note that for postdominators it automatically takes care of inserting
  /// a reverse edge internally (so there's no need to swap the parameters).
  ///
  void insertEdge(NodeT *From, NodeT *To) {
    assert(From);
    assert(To);
    assert(From->getParent() == Parent);
    assert(To->getParent() == Parent);
    DomTreeBuilder::InsertEdge(*this, From, To);
  }

  /// Inform the dominator tree about a CFG edge deletion and update the tree.
  ///
  /// This function has to be called just after making the update on the actual
  /// CFG. An internal functions checks if the edge doesn't exist in the CFG in
  /// DEBUG mode. There cannot be any other updates that the
  /// dominator tree doesn't know about.
  ///
  /// Note that for postdominators it automatically takes care of deleting
  /// a reverse edge internally (so there's no need to swap the parameters).
  ///
  void deleteEdge(NodeT *From, NodeT *To) {
    assert(From);
    assert(To);
    assert(From->getParent() == Parent);
    assert(To->getParent() == Parent);
    DomTreeBuilder::DeleteEdge(*this, From, To);
  }

  /// Add a new node to the dominator tree information.
  ///
  /// This creates a new node as a child of DomBB dominator node, linking it
  /// into the children list of the immediate dominator.
  ///
  /// \param BB New node in CFG.
  /// \param DomBB CFG node that is dominator for BB.
  /// \returns New dominator tree node that represents new CFG node.
  ///
  DomTreeNodeBase<NodeT> *addNewBlock(NodeT *BB, NodeT *DomBB) {
    assert(getNode(BB) == nullptr && "Block already in dominator tree!");
    DomTreeNodeBase<NodeT> *IDomNode = getNode(DomBB);
    assert(IDomNode && "Not immediate dominator specified for block!");
    DFSInfoValid = false;
    return createChild(BB, IDomNode);
  }

  /// Add a new node to the forward dominator tree and make it a new root.
  ///
  /// \param BB New node in CFG.
  /// \returns New dominator tree node that represents new CFG node.
  ///
  DomTreeNodeBase<NodeT> *setNewRoot(NodeT *BB) {
    assert(getNode(BB) == nullptr && "Block already in dominator tree!");
    assert(!this->isPostDominator() &&
           "Cannot change root of post-dominator tree");
    DFSInfoValid = false;
    DomTreeNodeBase<NodeT> *NewNode = createNode(BB);
    if (Roots.empty()) {
      addRoot(BB);
    } else {
      assert(Roots.size() == 1);
      NodeT *OldRoot = Roots.front();
      auto &OldNode = DomTreeNodes[CfgTraits::wrapRef(OldRoot)];
      OldNode = NewNode->addChild(std::move(OldNode));
      OldNode->IDom = NewNode;
      OldNode->UpdateLevel();
      Roots[0] = BB;
    }
    RootNode = NewNode;
    return static_cast<DomTreeNodeBase<NodeT> *>(RootNode);
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
    assert(Node->isLeaf() && "Node is not a leaf node.");

    DFSInfoValid = false;

    // Remove node from immediate dominator's children list.
    DomTreeNodeBase<NodeT> *IDom = Node->getIDom();
    if (IDom) {
      const auto I = find(IDom->Children, Node);
      assert(I != IDom->Children.end() &&
             "Not in immediate dominator children set!");
      // I am no longer your child...
      IDom->Children.erase(I);
    }

    DomTreeNodes.erase(CfgTraits::wrapRef(BB));

    if (!IsPostDom) return;

    // Remember to update PostDominatorTree roots.
    auto RIt = llvm::find(Roots, BB);
    if (RIt != Roots.end()) {
      std::swap(*RIt, Roots.back());
      Roots.pop_back();
    }
  }

  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  void splitBlock(NodeT *NewBB) {
    if (IsPostDominator)
      Split<Inverse<NodeT *>>(NewBB);
    else
      Split<NodeT *>(NewBB);
  }

  /// print - Convert to human readable form
  ///
  void print(raw_ostream &O) const {
    O << "=============================--------------------------------\n";
    if (IsPostDominator)
      O << "Inorder PostDominator Tree: ";
    else
      O << "Inorder Dominator Tree: ";
    if (!DFSInfoValid)
      O << "DFSNumbers invalid: " << SlowQueries << " slow queries.";
    O << "\n";

    // The postdom tree can have a null root if there are no returns.
    if (getRootNode()) PrintDomTree<NodeT>(getRootNode(), O, 1);
    O << "Roots: ";
    for (const NodePtr Block : Roots) {
      Block->printAsOperand(O, false);
      O << " ";
    }
    O << "\n";
  }

public:
  /// recalculate - compute a dominator tree for the given function
  void recalculate(ParentType &Func) {
    Parent = &Func;
    DomTreeBuilder::Calculate(*this);
  }

  void recalculate(ParentType &Func, ArrayRef<UpdateType> Updates) {
    Parent = &Func;
    DomTreeBuilder::CalculateWithUpdates(*this, Updates);
  }

  /// verify - checks if the tree is correct. There are 3 level of verification:
  ///  - Full --  verifies if the tree is correct by making sure all the
  ///             properties (including the parent and the sibling property)
  ///             hold.
  ///             Takes O(N^3) time.
  ///
  ///  - Basic -- checks if the tree is correct, but compares it to a freshly
  ///             constructed tree instead of checking the sibling property.
  ///             Takes O(N^2) time.
  ///
  ///  - Fast  -- checks basic tree structure and compares it with a freshly
  ///             constructed tree.
  ///             Takes O(N^2) time worst case, but is faster in practise (same
  ///             as tree construction).
  bool verify(VerificationLevel VL = VerificationLevel::Full) const {
    return DomTreeBuilder::Verify(*this, VL);
  }

  void reset() {
    GenericDominatorTreeBase::reset();
    Roots.clear();
    Parent = nullptr;
  }

protected:
  void addRoot(NodeT *BB) { this->Roots.push_back(BB); }

  DomTreeNodeBase<NodeT> *createChild(NodeT *BB, DomTreeNodeBase<NodeT> *IDom) {
    CfgBlockRef bbRef = CfgTraits::wrapRef(BB);
    return static_cast<DomTreeNodeBase<NodeT> *>(
        (DomTreeNodes[bbRef] = IDom->addChild(
             std::make_unique<GenericDomTreeNodeBase>(bbRef, IDom)))
            .get());
  }

  DomTreeNodeBase<NodeT> *createNode(NodeT *BB) {
    CfgBlockRef bbRef = CfgTraits::wrapRef(BB);
    return static_cast<DomTreeNodeBase<NodeT> *>(
        (DomTreeNodes[bbRef] =
             std::make_unique<GenericDomTreeNodeBase>(bbRef, nullptr))
            .get());
  }

  // NewBB is split and now it has one successor. Update dominator tree to
  // reflect this change.
  template <class N>
  void Split(typename GraphTraits<N>::NodeRef NewBB) {
    using GraphT = GraphTraits<N>;
    using NodeRef = typename GraphT::NodeRef;
    assert(std::distance(GraphT::child_begin(NewBB),
                         GraphT::child_end(NewBB)) == 1 &&
           "NewBB should have a single successor!");
    NodeRef NewBBSucc = *GraphT::child_begin(NewBB);

    SmallVector<NodeRef, 4> PredBlocks;
    for (auto Pred : children<Inverse<N>>(NewBB))
      PredBlocks.push_back(Pred);

    assert(!PredBlocks.empty() && "No predblocks?");

    bool NewBBDominatesNewBBSucc = true;
    for (auto Pred : children<Inverse<N>>(NewBBSucc)) {
      if (Pred != NewBB && !dominates(NewBBSucc, Pred) &&
          isReachableFromEntry(Pred)) {
        NewBBDominatesNewBBSucc = false;
        break;
      }
    }

    // Find NewBB's immediate dominator and create new dominator tree node for
    // NewBB.
    NodeT *NewBBIDom = nullptr;
    unsigned i = 0;
    for (i = 0; i < PredBlocks.size(); ++i)
      if (isReachableFromEntry(PredBlocks[i])) {
        NewBBIDom = PredBlocks[i];
        break;
      }

    // It's possible that none of the predecessors of NewBB are reachable;
    // in that case, NewBB itself is unreachable, so nothing needs to be
    // changed.
    if (!NewBBIDom) return;

    for (i = i + 1; i < PredBlocks.size(); ++i) {
      if (isReachableFromEntry(PredBlocks[i]))
        NewBBIDom = findNearestCommonDominator(NewBBIDom, PredBlocks[i]);
    }

    // Create the new dominator tree node... and set the idom of NewBB.
    DomTreeNodeBase<NodeT> *NewBBNode = addNewBlock(NewBB, NewBBIDom);

    // If NewBB strictly dominates other blocks, then it is now the immediate
    // dominator of NewBBSucc.  Update the dominator tree as appropriate.
    if (NewBBDominatesNewBBSucc) {
      DomTreeNodeBase<NodeT> *NewBBSuccNode = getNode(NewBBSucc);
      changeImmediateDominator(NewBBSuccNode, NewBBNode);
    }
  }
};

template <typename T>
using DomTreeBase = DominatorTreeBase<T, false>;

template <typename T>
using PostDomTreeBase = DominatorTreeBase<T, true>;

} // end namespace llvm

#endif // LLVM_SUPPORT_GENERICDOMTREE_H
