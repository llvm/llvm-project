//===--- ImmutableSet.h - Immutable (functional) set interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ImutAVLTree and ImmutableSet classes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMMUTABLESET_H
#define LLVM_ADT_IMMUTABLESET_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <new>
#include <vector>

namespace llvm {

//===----------------------------------------------------------------------===//
// Immutable AVL-Tree Definition.
//===----------------------------------------------------------------------===//

template <typename ImutInfo, bool Canonicalize = true> class ImutAVLFactory;
template <typename ImutInfo> class ImutIntervalAVLFactory;
template <typename ImutInfo, bool Canonicalize = true>
class ImutAVLTreeInOrderIterator;

namespace ImutAVLDetail {
/// The intrusive doubly-linked chain of same-digest trees in the factory's
/// canonicalization cache. Held as an (empty) base so that, when
/// canonicalization is disabled, the empty base optimization removes it
/// entirely. Kept separate from the cached digest below so that the two
/// pointers pack without the tail padding that grouping a trailing 32-bit field
/// with them would introduce.
template <typename Tree, bool Canonicalize> struct CanonicalLinks {
  Tree *Prev = nullptr;
  Tree *Next = nullptr;
};
template <typename Tree> struct CanonicalLinks<Tree, false> {};

/// The cached structural digest, used only for canonicalization. Stored as an
/// LLVM_NO_UNIQUE_ADDRESS member so it occupies no space when disabled and
/// packs alongside the adjacent 32-bit fields when enabled.
template <bool Canonicalize> struct CanonicalDigest {
  uint32_t Digest = 0;
};
template <> struct CanonicalDigest<false> {};

/// The factory-side canonicalization cache: digest -> tree chain.
/// Empty when canonicalization is disabled.
template <typename Tree, bool Canonicalize> struct CanonicalCache {
  DenseMap<unsigned, Tree *> Cache;
};
template <typename Tree> struct CanonicalCache<Tree, false> {};
} // namespace ImutAVLDetail

template <typename ImutInfo, bool Canonicalize = true>
class ImutAVLTree
    : private ImutAVLDetail::CanonicalLinks<ImutAVLTree<ImutInfo, Canonicalize>,
                                            Canonicalize> {
public:
  using key_type_ref = typename ImutInfo::key_type_ref;
  using value_type = typename ImutInfo::value_type;
  using value_type_ref = typename ImutInfo::value_type_ref;
  using Factory = ImutAVLFactory<ImutInfo, Canonicalize>;
  using iterator = ImutAVLTreeInOrderIterator<ImutInfo, Canonicalize>;

  friend class ImutAVLFactory<ImutInfo, Canonicalize>;
  friend class ImutIntervalAVLFactory<ImutInfo>;

private:
  using CanonLinks = ImutAVLDetail::CanonicalLinks<ImutAVLTree, Canonicalize>;

public:
  //===----------------------------------------------------===//
  // Public Interface.
  //===----------------------------------------------------===//

  /// Return a pointer to the left subtree.  This value
  ///  is NULL if there is no left subtree.
  ImutAVLTree *getLeft() const { return left; }

  /// Return a pointer to the right subtree.  This value is
  ///  NULL if there is no right subtree.
  ImutAVLTree *getRight() const { return right; }

  /// Returns the height of the tree. A tree with no subtrees has a height of 1.
  unsigned getHeight() const { return height; }

  /// Returns the data value associated with the tree node.
  const value_type& getValue() const { return value; }

  /// Finds the subtree associated with the specified key value. This method
  /// returns NULL if no matching subtree is found.
  ImutAVLTree* find(key_type_ref K) {
    ImutAVLTree *T = this;
    while (T) {
      key_type_ref CurrentKey = ImutInfo::KeyOfValue(T->getValue());
      if (ImutInfo::isEqual(K,CurrentKey))
        return T;
      else if (ImutInfo::isLess(K,CurrentKey))
        T = T->getLeft();
      else
        T = T->getRight();
    }
    return nullptr;
  }

  /// Find the subtree associated with the highest ranged key value.
  ImutAVLTree* getMaxElement() {
    ImutAVLTree *T = this;
    ImutAVLTree *Right = T->getRight();
    while (Right) { T = Right; Right = T->getRight(); }
    return T;
  }

  /// Returns the number of nodes in the tree, which includes both leaves and
  // non-leaf nodes.
  unsigned size() const {
    unsigned n = 1;
    if (const ImutAVLTree* L = getLeft())
      n += L->size();
    if (const ImutAVLTree* R = getRight())
      n += R->size();
    return n;
  }

  /// Returns an iterator that iterates over the nodes of the tree in an inorder
  /// traversal. The returned iterator thus refers to the tree node with the
  /// minimum data element.
  iterator begin() const { return iterator(this); }

  /// Returns an iterator for the tree that denotes the end of an inorder
  /// traversal.
  iterator end() const { return iterator(); }

  bool isElementEqual(value_type_ref V) const {
    // Compare the keys.
    if (!ImutInfo::isEqual(ImutInfo::KeyOfValue(getValue()),
                           ImutInfo::KeyOfValue(V)))
      return false;

    // Also compare the data values.
    if (!ImutInfo::isDataEqual(ImutInfo::DataOfValue(getValue()),
                               ImutInfo::DataOfValue(V)))
      return false;

    return true;
  }

  bool isElementEqual(const ImutAVLTree* RHS) const {
    return isElementEqual(RHS->getValue());
  }

  /// Compares two trees for structural equality and returns true if they are
  /// equal. The worst case performance of this operation is linear in the sizes
  /// of the trees.
  bool isEqual(const ImutAVLTree& RHS) const {
    if (&RHS == this)
      return true;

    iterator LItr = begin(), LEnd = end();
    iterator RItr = RHS.begin(), REnd = RHS.end();

    while (LItr != LEnd && RItr != REnd) {
      if (&*LItr == &*RItr) {
        LItr.skipSubTree();
        RItr.skipSubTree();
        continue;
      }

      if (!LItr->isElementEqual(&*RItr))
        return false;

      ++LItr;
      ++RItr;
    }

    return LItr == LEnd && RItr == REnd;
  }

  /// Compares two trees for structural inequality.  Performance is the same as
  /// isEqual.
  bool isNotEqual(const ImutAVLTree& RHS) const { return !isEqual(RHS); }

  /// Returns true if this tree contains a subtree (node) that has an data
  /// element that matches the specified key. Complexity is logarithmic in the
  /// size of the tree.
  bool contains(key_type_ref K) { return (bool) find(K); }

  /// A utility method that checks that the balancing and ordering invariants of
  /// the tree are satisfied. It is a recursive method that returns the height
  /// of the tree, which is then consumed by the enclosing validateTree call.
  /// External callers should ignore the return value.  An invalid tree will
  /// cause an assertion to fire in a debug build.
  unsigned validateTree() const {
    unsigned HL = getLeft() ? getLeft()->validateTree() : 0;
    unsigned HR = getRight() ? getRight()->validateTree() : 0;
    (void) HL;
    (void) HR;

    assert(getHeight() == ( HL > HR ? HL : HR ) + 1
            && "Height calculation wrong");

    assert((HL > HR ? HL-HR : HR-HL) <= 2
           && "Balancing invariant violated");

    assert((!getLeft() ||
            ImutInfo::isLess(ImutInfo::KeyOfValue(getLeft()->getValue()),
                             ImutInfo::KeyOfValue(getValue()))) &&
           "Value in left child is not less that current value");

    assert((!getRight() ||
             ImutInfo::isLess(ImutInfo::KeyOfValue(getValue()),
                              ImutInfo::KeyOfValue(getRight()->getValue()))) &&
           "Current value is not less that value of right child");

    return getHeight();
  }

  //===----------------------------------------------------===//
  // Internal values.
  //===----------------------------------------------------===//

private:
  // Field order places the traversal-hot fields (left, right, value) first so
  // that in a node that straddles a cache line they land in the earlier line;
  // the cold factory back-pointer (only touched on create/destroy) goes last.
  ImutAVLTree *left;
  ImutAVLTree *right;

  unsigned height : 28;
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsMutable : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsDigestCached : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsCanonicalized : 1;

  value_type value;
  uint32_t refCount = 0;
  LLVM_NO_UNIQUE_ADDRESS ImutAVLDetail::CanonicalDigest<Canonicalize> digest;
  Factory *factory;

  //===----------------------------------------------------===//
  // Internal methods (node manipulation; used by Factory).
  //===----------------------------------------------------===//

private:
  /// Internal constructor that is only called by ImutAVLFactory.
  ImutAVLTree(Factory *f, ImutAVLTree *l, ImutAVLTree *r, value_type_ref v,
              unsigned height)
      : left(l), right(r), height(height), IsMutable(true),
        IsDigestCached(false), IsCanonicalized(false), value(v), factory(f) {
    if (left) left->retain();
    if (right) right->retain();
  }

  /// Returns true if the left and right subtree references
  ///  (as well as height) can be changed.  If this method returns false,
  ///  the tree is truly immutable.  Trees returned from an ImutAVLFactory
  ///  object should always have this method return true.  Further, if this
  ///  method returns false for an instance of ImutAVLTree, all subtrees
  ///  will also have this method return false.  The converse is not true.
  bool isMutable() const { return IsMutable; }

  /// Returns true if the digest for this tree is cached. This can only be true
  /// if the tree is immutable.
  bool hasCachedDigest() const { return IsDigestCached; }

  //===----------------------------------------------------===//
  // Mutating operations.  A tree root can be manipulated as
  // long as its reference has not "escaped" from internal
  // methods of a factory object (see below).  When a tree
  // pointer is externally viewable by client code, the
  // internal "mutable bit" is cleared to mark the tree
  // immutable.  Note that a tree that still has its mutable
  // bit set may have children (subtrees) that are themselves
  // immutable.
  //===----------------------------------------------------===//

  /// Clears the mutable flag for a tree.  After this happens,
  /// it is an error to call setLeft(), setRight(), and setHeight().
  void markImmutable() {
    assert(isMutable() && "Mutable flag already removed.");
    IsMutable = false;
  }

  /// Clears the NoCachedDigest flag for a tree.
  void markedCachedDigest() {
    assert(!hasCachedDigest() && "NoCachedDigest flag already removed.");
    IsDigestCached = true;
  }

  /// Changes the height of the tree.  Used internally by ImutAVLFactory.
  void setHeight(unsigned h) {
    assert(isMutable() && "Only a mutable tree can have its height changed.");
    height = h;
  }

  static uint32_t computeDigest(ImutAVLTree *L, ImutAVLTree *R,
                                value_type_ref V) {
    uint32_t digest = 0;

    if (L)
      digest += L->computeDigest();

    // Compute digest of stored data.
    FoldingSetNodeID ID;
    ImutInfo::Profile(ID,V);
    digest += ID.ComputeHash();

    if (R)
      digest += R->computeDigest();

    return digest;
  }

  uint32_t computeDigest() {
    // Check the lowest bit to determine if digest has actually been
    // pre-computed.
    if (hasCachedDigest())
      return digest.Digest;

    uint32_t X = computeDigest(getLeft(), getRight(), getValue());
    digest.Digest = X;
    markedCachedDigest();
    return X;
  }

  //===----------------------------------------------------===//
  // Reference count operations.
  //===----------------------------------------------------===//

public:
  void retain() { ++refCount; }

  void release() {
    assert(refCount > 0);
    if (--refCount == 0)
      destroy();
  }

  LLVM_ATTRIBUTE_NOINLINE void destroy() {
    if (left)
      left->release();
    if (right)
      right->release();
    if constexpr (Canonicalize) {
      if (IsCanonicalized) {
        if (this->Next)
          this->Next->Prev = this->Prev;

        if (this->Prev)
          this->Prev->Next = this->Next;
        else
          factory->Cache[factory->maskCacheIndex(computeDigest())] = this->Next;
      }
    }

    // We need to clear the mutability bit in case we are
    // destroying the node as part of a sweep in ImutAVLFactory::recoverNodes().
    IsMutable = false;
    factory->freeNodes.push_back(this);
  }
};

template <typename ImutInfo, bool Canonicalize>
struct IntrusiveRefCntPtrInfo<ImutAVLTree<ImutInfo, Canonicalize>> {
  static void retain(ImutAVLTree<ImutInfo, Canonicalize> *Tree) {
    Tree->retain();
  }
  static void release(ImutAVLTree<ImutInfo, Canonicalize> *Tree) {
    Tree->release();
  }
};

//===----------------------------------------------------------------------===//
// Immutable AVL-Tree Factory class.
//===----------------------------------------------------------------------===//

template <typename ImutInfo, bool Canonicalize>
class ImutAVLFactory
    : private ImutAVLDetail::CanonicalCache<ImutAVLTree<ImutInfo, Canonicalize>,
                                            Canonicalize> {
  friend class ImutAVLTree<ImutInfo, Canonicalize>;

  using TreeTy = ImutAVLTree<ImutInfo, Canonicalize>;
  using value_type = typename TreeTy::value_type;
  using value_type_ref = typename TreeTy::value_type_ref;
  using key_type_ref = typename TreeTy::key_type_ref;

  uintptr_t Allocator;
  std::vector<TreeTy*> createdNodes;
  std::vector<TreeTy*> freeNodes;

  bool ownsAllocator() const {
    return (Allocator & 0x1) == 0;
  }

  BumpPtrAllocator& getAllocator() const {
    return *reinterpret_cast<BumpPtrAllocator*>(Allocator & ~0x1);
  }

  //===--------------------------------------------------===//
  // Public interface.
  //===--------------------------------------------------===//

public:
  ImutAVLFactory()
    : Allocator(reinterpret_cast<uintptr_t>(new BumpPtrAllocator())) {}

  ImutAVLFactory(BumpPtrAllocator& Alloc)
    : Allocator(reinterpret_cast<uintptr_t>(&Alloc) | 0x1) {}

  ~ImutAVLFactory() {
    if (ownsAllocator()) delete &getAllocator();
  }

  TreeTy* add(TreeTy* T, value_type_ref V) {
    T = add_internal(V,T);
    recoverNodes(T);
    return T;
  }

  /// Merges \p A and \p B in a single traversal, sharing every subtree that the
  /// two operands do not overlap. \p Combine(AElem, BElem) produces the element
  /// stored for a key present in both; \p KeepUnmatched governs keys unique to
  /// one side (see merge_internal). For merging |B| entries into |A|
  /// (|B| <= |A|) this costs O(|B| * log(|A|/|B| + 1)) and copies each spine
  /// node at most once, versus O(|B| * log|A|) repeated \ref add descents.
  /// \p A and \p B must be immutable. This does not short-circuit equal or
  /// empty operands (merge_internal handles them correctly but not specially);
  /// callers that want those fast paths, or size-driven operand ordering,
  /// should apply them first (see ImmutableSet::Factory::unionSets).
  template <typename CombineFn>
  TreeTy *mergeTrees(TreeTy *A, TreeTy *B, CombineFn Combine,
                     bool KeepUnmatched, bool SkipShared = false) {
    TreeTy *T = merge_internal(A, B, Combine, KeepUnmatched, SkipShared);
    recoverNodes(T);
    return T;
  }

  /// Returns the set union of \p A and \p B (keeping \p A's element on matching
  /// keys). Shorthand for the fully sharing \ref mergeTrees.
  TreeTy *unionTrees(TreeTy *A, TreeTy *B) {
    // With KeepUnmatched=true, unmatched elements are shared as-is and Combine
    // is invoked only for keys present in both, where it keeps A's element.
    auto KeepFirst = [](const value_type *L,
                        const value_type *R) -> const value_type & {
      return L ? *L : *R;
    };
    // Set union is idempotent, so identical (pointer-equal) subtrees -- common
    // once one operand is derived from the other -- can be shared in O(1).
    return mergeTrees(A, B, KeepFirst, /*KeepUnmatched=*/true,
                      /*SkipShared=*/true);
  }

  TreeTy* remove(TreeTy* T, key_type_ref V) {
    T = remove_internal(V,T);
    recoverNodes(T);
    return T;
  }

  TreeTy* getEmptyTree() const { return nullptr; }

protected:
  //===--------------------------------------------------===//
  // A bunch of quick helper functions used for reasoning
  // about the properties of trees and their children.
  // These have succinct names so that the balancing code
  // is as terse (and readable) as possible.
  //===--------------------------------------------------===//

  bool            isEmpty(TreeTy* T) const { return !T; }
  unsigned        getHeight(TreeTy* T) const { return T ? T->getHeight() : 0; }
  TreeTy*         getLeft(TreeTy* T) const { return T->getLeft(); }
  TreeTy*         getRight(TreeTy* T) const { return T->getRight(); }
  value_type_ref  getValue(TreeTy* T) const { return T->value; }

  // Make sure the index is not the Tombstone or Entry key of the DenseMap.
  static unsigned maskCacheIndex(unsigned I) { return (I & ~0x02); }

  unsigned incrementHeight(TreeTy* L, TreeTy* R) const {
    unsigned hl = getHeight(L);
    unsigned hr = getHeight(R);
    return (hl > hr ? hl : hr) + 1;
  }

  //===--------------------------------------------------===//
  // "createNode" is used to generate new tree roots that link
  // to other trees.  The function may also simply move links
  // in an existing root if that root is still marked mutable.
  // This is necessary because otherwise our balancing code
  // would leak memory as it would create nodes that are
  // then discarded later before the finished tree is
  // returned to the caller.
  //===--------------------------------------------------===//

  TreeTy* createNode(TreeTy* L, value_type_ref V, TreeTy* R) {
    BumpPtrAllocator& A = getAllocator();
    TreeTy* T;
    if (!freeNodes.empty()) {
      T = freeNodes.back();
      freeNodes.pop_back();
      assert(T != L);
      assert(T != R);
    } else {
      T = (TreeTy*) A.Allocate<TreeTy>();
    }
    new (T) TreeTy(this, L, R, V, incrementHeight(L,R));
    createdNodes.push_back(T);
    return T;
  }

  TreeTy* createNode(TreeTy* newLeft, TreeTy* oldTree, TreeTy* newRight) {
    return createNode(newLeft, getValue(oldTree), newRight);
  }

  void recoverNodes(TreeTy *Result) {
    // Mark Result's nodes immutable and reclaim the intermediates discarded
    // during balancing, in one pass. Nodes are built bottom-up, so a node
    // precedes its parents in createdNodes; visiting in reverse thus reaches
    // each node only once its reference count is final. Unreferenced nodes are
    // unreachable and destroyed; the rest belong to Result. Result is kept
    // despite its zero count -- the caller has not taken ownership yet.
    for (TreeTy *N : llvm::reverse(createdNodes)) {
      if (!N->isMutable())
        continue; // Already reclaimed while destroying an unreachable parent.
      if (N != Result && N->refCount == 0)
        N->destroy();
      else
        N->markImmutable();
    }
    createdNodes.clear();
  }

  /// Used by add_internal and remove_internal to balance a newly created tree.
  TreeTy* balanceTree(TreeTy* L, value_type_ref V, TreeTy* R) {
    unsigned hl = getHeight(L);
    unsigned hr = getHeight(R);

    if (hl > hr + 2) {
      assert(!isEmpty(L) && "Left tree cannot be empty to have a height >= 2");

      TreeTy *LL = getLeft(L);
      TreeTy *LR = getRight(L);

      if (getHeight(LL) >= getHeight(LR))
        return createNode(LL, L, createNode(LR,V,R));

      assert(!isEmpty(LR) && "LR cannot be empty because it has a height >= 1");

      TreeTy *LRL = getLeft(LR);
      TreeTy *LRR = getRight(LR);

      return createNode(createNode(LL,L,LRL), LR, createNode(LRR,V,R));
    }

    if (hr > hl + 2) {
      assert(!isEmpty(R) && "Right tree cannot be empty to have a height >= 2");

      TreeTy *RL = getLeft(R);
      TreeTy *RR = getRight(R);

      if (getHeight(RR) >= getHeight(RL))
        return createNode(createNode(L,V,RL), R, RR);

      assert(!isEmpty(RL) && "RL cannot be empty because it has a height >= 1");

      TreeTy *RLL = getLeft(RL);
      TreeTy *RLR = getRight(RL);

      return createNode(createNode(L,V,RLL), RL, createNode(RLR,R,RR));
    }

    return createNode(L,V,R);
  }

  /// Combines \p L and \p R with the value \p V (every key in \p L less than
  /// \p V, every key in \p R greater) into one balanced tree. Unlike
  /// balanceTree this tolerates an arbitrary height difference between \p L and
  /// \p R: it descends the taller side's spine and rebalances on the way back
  /// up, exactly as an insertion would.
  TreeTy *joinTrees(TreeTy *L, value_type_ref V, TreeTy *R) {
    if (getHeight(L) > getHeight(R) + 2)
      return balanceTree(getLeft(L), getValue(L), joinTrees(getRight(L), V, R));
    if (getHeight(R) > getHeight(L) + 2)
      return balanceTree(joinTrees(L, V, getLeft(R)), getValue(R), getRight(R));
    return createNode(L, V, R);
  }

  /// Splits \p T into \p L (all keys less than \p K) and \p R (all keys greater
  /// than \p K). If \p K is present in \p T, \p Match is set to point at its
  /// element (which is dropped from \p L and \p R); otherwise \p Match is null.
  void splitLookup(TreeTy *T, key_type_ref K, TreeTy *&L,
                   const value_type *&Match, TreeTy *&R) {
    if (isEmpty(T)) {
      L = R = getEmptyTree();
      Match = nullptr;
      return;
    }
    key_type_ref KCurrent = ImutInfo::KeyOfValue(getValue(T));
    if (ImutInfo::isEqual(K, KCurrent)) {
      L = getLeft(T);
      R = getRight(T);
      // Use the tree accessor, which returns a reference to the stored element
      // (the factory's getValue returns value_type_ref, which is by value for
      // pointer-like element types).
      Match = &T->getValue();
    } else if (ImutInfo::isLess(K, KCurrent)) {
      TreeTy *LR;
      splitLookup(getLeft(T), K, L, Match, LR);
      R = joinTrees(LR, getValue(T), getRight(T));
    } else {
      TreeTy *RL;
      splitLookup(getRight(T), K, RL, Match, R);
      L = joinTrees(getLeft(T), getValue(T), RL);
    }
  }

  /// Rebuilds \p T with the same shape but each element replaced by
  /// \p Combine applied to it. \p FromB selects which side of \p Combine the
  /// element is passed on (it is the sole non-null argument).
  template <typename CombineFn>
  TreeTy *transformTree(TreeTy *T, CombineFn &Combine, bool FromB) {
    if (isEmpty(T))
      return T;
    TreeTy *L = transformTree(getLeft(T), Combine, FromB);
    TreeTy *R = transformTree(getRight(T), Combine, FromB);
    const value_type &E = getValue(T);
    value_type NewE = FromB ? Combine(nullptr, &E) : Combine(&E, nullptr);
    if (L == getLeft(T) && R == getRight(T) && T->isElementEqual(NewE))
      return T;
    return createNode(L, NewE, R);
  }

  /// Merges \p A and \p B by recursing over \p A's structure and splitting \p B
  /// at each of \p A's keys. For a key in both, the stored element is
  /// Combine(AElem, BElem). \p KeepUnmatched controls keys unique to one side:
  /// when true, such elements (and whole non-overlapping subtrees) are taken
  /// unchanged and shared, and \p Combine is invoked only on keys present in
  /// both (valid when \p Combine is an identity for a missing side, e.g. a set
  /// union or a lattice join with an identity element); when false every key is
  /// passed through \p Combine with the absent side null (needed for a join
  /// that transforms unmatched keys, e.g. liveness downgrading Must to Maybe).
  template <typename CombineFn>
  TreeTy *merge_internal(TreeTy *A, TreeTy *B, CombineFn &Combine,
                         bool KeepUnmatched, bool SkipShared) {
    // When A and B are the same tree (which happens all the time once B is
    // derived from A by a small edit, since the untouched side is shared by
    // pointer), an idempotent merge returns it unchanged in O(1). Only valid
    // when merge(x, x) == x, so the caller opts in via SkipShared.
    if (SkipShared && A == B)
      return A;
    if (isEmpty(A))
      return KeepUnmatched ? B : transformTree(B, Combine, /*FromB=*/true);
    if (isEmpty(B))
      return KeepUnmatched ? A : transformTree(A, Combine, /*FromB=*/false);

    const value_type &AElem = getValue(A);
    TreeTy *BL, *BR;
    const value_type *BMatch;
    splitLookup(B, ImutInfo::KeyOfValue(AElem), BL, BMatch, BR);

    TreeTy *NewL =
        merge_internal(getLeft(A), BL, Combine, KeepUnmatched, SkipShared);
    TreeTy *NewR =
        merge_internal(getRight(A), BR, Combine, KeepUnmatched, SkipShared);

    if (!BMatch) {
      // Key present only in A.
      if (KeepUnmatched) {
        if (NewL == getLeft(A) && NewR == getRight(A))
          return A;
        return joinTrees(NewL, AElem, NewR);
      }
      auto NewE = Combine(&AElem, nullptr);
      if (NewL == getLeft(A) && NewR == getRight(A) && A->isElementEqual(NewE))
        return A;
      return joinTrees(NewL, NewE, NewR);
    }
    // Key present in both: combine the two elements. Preserve sharing when the
    // combined value is unchanged and neither subtree moved, so that a join
    // that only touches a few keys does not rebuild the whole spine.
    auto NewElem = Combine(&AElem, BMatch);
    if (NewL == getLeft(A) && NewR == getRight(A) &&
        ImutInfo::isDataEqual(ImutInfo::DataOfValue(NewElem),
                              ImutInfo::DataOfValue(AElem)))
      return A;
    return joinTrees(NewL, NewElem, NewR);
  }

  /// add_internal - Creates a new tree that includes the specified
  ///  data and the data from the original tree.  If the original tree
  ///  already contained the data item, the original tree is returned.
  TreeTy *add_internal(value_type_ref V, TreeTy *T) {
    if (isEmpty(T))
      return createNode(T, V, T);
    assert(!T->isMutable());

    key_type_ref K = ImutInfo::KeyOfValue(V);
    key_type_ref KCurrent = ImutInfo::KeyOfValue(getValue(T));

    if (ImutInfo::isEqual(K, KCurrent)) {
      // If both key and value are same, return the original tree.
      if (ImutInfo::isDataEqual(ImutInfo::DataOfValue(V),
                                ImutInfo::DataOfValue(getValue(T))))
        return T;
      // Otherwise create a new node with the new value.
      return createNode(getLeft(T), V, getRight(T));
    }

    TreeTy *NewL = getLeft(T);
    TreeTy *NewR = getRight(T);
    if (ImutInfo::isLess(K, KCurrent))
      NewL = add_internal(V, NewL);
    else
      NewR = add_internal(V, NewR);

    // If no changes were made, return the original tree. Otherwise, balance the
    // tree and return the new root.
    return NewL == getLeft(T) && NewR == getRight(T)
               ? T
               : balanceTree(NewL, getValue(T), NewR);
  }

  /// remove_internal - Creates a new tree that includes all the data
  ///  from the original tree except the specified data.  If the
  ///  specified data did not exist in the original tree, the original
  ///  tree is returned.
  TreeTy *remove_internal(key_type_ref K, TreeTy *T) {
    if (isEmpty(T))
      return T;

    assert(!T->isMutable());

    key_type_ref KCurrent = ImutInfo::KeyOfValue(getValue(T));

    if (ImutInfo::isEqual(K, KCurrent))
      return combineTrees(getLeft(T), getRight(T));

    TreeTy *NewL = getLeft(T);
    TreeTy *NewR = getRight(T);
    if (ImutInfo::isLess(K, KCurrent))
      NewL = remove_internal(K, NewL);
    else
      NewR = remove_internal(K, NewR);

    // If no changes were made, return the original tree. Otherwise, balance the
    // tree and return the new root.
    return NewL == getLeft(T) && NewR == getRight(T)
               ? T
               : balanceTree(NewL, getValue(T), NewR);
  }

  TreeTy* combineTrees(TreeTy* L, TreeTy* R) {
    if (isEmpty(L))
      return R;
    if (isEmpty(R))
      return L;
    TreeTy* OldNode;
    TreeTy* newRight = removeMinBinding(R,OldNode);
    return balanceTree(L, getValue(OldNode), newRight);
  }

  TreeTy* removeMinBinding(TreeTy* T, TreeTy*& Noderemoved) {
    assert(!isEmpty(T));
    if (isEmpty(getLeft(T))) {
      Noderemoved = T;
      return getRight(T);
    }
    return balanceTree(removeMinBinding(getLeft(T), Noderemoved),
                       getValue(T), getRight(T));
  }

public:
  TreeTy *getCanonicalTree(TreeTy *TNew) {
    static_assert(Canonicalize,
                  "getCanonicalTree requires a canonicalizing factory");
    if (!TNew)
      return nullptr;

    if (TNew->IsCanonicalized)
      return TNew;

    // Search the hashtable for another tree with the same digest, and
    // if find a collision compare those trees by their contents.
    unsigned digest = TNew->computeDigest();
    TreeTy *&entry = this->Cache[maskCacheIndex(digest)];
    if (entry) {
      for (TreeTy *T = entry; T != nullptr; T = T->Next) {
        // Compare the contents of 'T' with 'TNew'. isEqual skips subtrees that
        // are shared by pointer, so for structurally-shared persistent trees
        // (the common case, e.g. one derived from the other) this is linear in
        // the number of differing nodes rather than in the tree size.
        if (!TNew->isEqual(*T))
          continue;
        // Trees did match!  Return 'T'.
        if (TNew->refCount == 0)
          TNew->destroy();
        return T;
      }
      entry->Prev = TNew;
      TNew->Next = entry;
    }

    entry = TNew;
    TNew->IsCanonicalized = true;
    return TNew;
  }
};

//===----------------------------------------------------------------------===//
// Immutable AVL-Tree Iterator.
//===----------------------------------------------------------------------===//

/// Bidirectional in-order iterator over the nodes of an ImutAVLTree.
///
/// The iterator keeps the chain of ancestors from the root down to the current
/// node on an explicit stack of plain node pointers, and decides which way to
/// move next by inspecting whether it is ascending from a node's left or right
/// child. This avoids storing any per-node visit-state: there is no need to
/// remember "have I already visited this node's left/right subtree", because
/// that is recovered by comparing the child we just left against the parent's
/// left and right pointers.
///
/// A node's parent cannot be cached in the node itself, because these trees are
/// persistent and structurally shared: a single node may appear as the child of
/// different parents across different tree versions. The ancestor stack is
/// therefore the per-traversal parent chain.
template <typename ImutInfo, bool Canonicalize>
class ImutAVLTreeInOrderIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = ImutAVLTree<ImutInfo, Canonicalize>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  using TreeTy = ImutAVLTree<ImutInfo, Canonicalize>;

private:
  // Path[0] is the root and Path.back() is the current node. An empty path is
  // the end iterator. The invariant is that Path always holds the exact chain
  // of ancestors of the current node, root-most first.
  SmallVector<TreeTy *, 20> Path;

  // Descend along left children, pushing each node; lands on the minimum of the
  // subtree rooted at T (i.e. the first node in an in-order traversal of T).
  void descendToMin(TreeTy *T) {
    for (; T; T = T->getLeft())
      Path.push_back(T);
  }

  // Descend along right children, pushing each node; lands on the maximum of
  // the subtree rooted at T (i.e. the last node in an in-order traversal of T).
  void descendToMax(TreeTy *T) {
    for (; T; T = T->getRight())
      Path.push_back(T);
  }

  // Pop the current node and ascend until we reach an ancestor from its *left*
  // child, i.e. the first ancestor whose subtree is not yet fully visited. That
  // ancestor is the in-order successor of the subtree we just left; if there is
  // none, Path is emptied (the end iterator). Shared by operator++ and
  // skipSubTree, whose only difference is whether the current node's right
  // subtree is descended into first.
  void ascendFromRightChild() {
    TreeTy *Child = Path.pop_back_val();
    while (!Path.empty() && Path.back()->getRight() == Child)
      Child = Path.pop_back_val();
  }

  // Mirror of ascendFromRightChild for reverse traversal (operator--).
  void ascendFromLeftChild() {
    TreeTy *Child = Path.pop_back_val();
    while (!Path.empty() && Path.back()->getLeft() == Child)
      Child = Path.pop_back_val();
  }

public:
  ImutAVLTreeInOrderIterator() = default; // end() iterator.
  ImutAVLTreeInOrderIterator(const TreeTy *Root) {
    descendToMin(const_cast<TreeTy *>(Root));
  }

  // Two iterators are equal iff they sit on the same node (or are both end()).
  // Within a single tree a node has a unique root-to-node path, so the current
  // node alone identifies the position; comparing the whole path is therefore
  // unnecessary. Comparing iterators from different trees is not meaningful, as
  // for any standard container.
  bool operator==(const ImutAVLTreeInOrderIterator &x) const {
    if (Path.empty() || x.Path.empty())
      return Path.empty() == x.Path.empty();
    return Path.back() == x.Path.back();
  }
  bool operator!=(const ImutAVLTreeInOrderIterator &x) const {
    return !(*this == x);
  }

  TreeTy &operator*() const { return *Path.back(); }
  TreeTy *operator->() const { return Path.back(); }

  ImutAVLTreeInOrderIterator &operator++() {
    assert(!Path.empty() && "Incrementing the end iterator");
    if (TreeTy *R = Path.back()->getRight())
      // The in-order successor is the minimum of the right subtree.
      descendToMin(R);
    else
      // No right subtree: the successor is the nearest ancestor reached from a
      // left child.
      ascendFromRightChild();
    return *this;
  }

  ImutAVLTreeInOrderIterator &operator--() {
    assert(!Path.empty() && "Decrementing the end iterator");
    if (TreeTy *L = Path.back()->getLeft())
      // The in-order predecessor is the maximum of the left subtree.
      descendToMax(L);
    else
      // Mirror of operator++.
      ascendFromLeftChild();
    return *this;
  }

  /// Move to the in-order successor of the entire subtree rooted at the current
  /// node, i.e. skip the current node together with its right subtree. This is
  /// exactly the ascent half of operator++.
  void skipSubTree() {
    assert(!Path.empty() && "Skipping past the end iterator");
    ascendFromRightChild();
  }
};

/// Generic iterator that wraps a T::TreeTy::iterator and exposes
/// iterator::getValue() on dereference.
template <typename T>
struct ImutAVLValueIterator
    : iterator_adaptor_base<
          ImutAVLValueIterator<T>, typename T::TreeTy::iterator,
          typename std::iterator_traits<
              typename T::TreeTy::iterator>::iterator_category,
          const typename T::value_type> {
  ImutAVLValueIterator() = default;
  explicit ImutAVLValueIterator(typename T::TreeTy *Tree)
      : ImutAVLValueIterator::iterator_adaptor_base(Tree) {}

  typename ImutAVLValueIterator::reference operator*() const {
    return this->I->getValue();
  }
};

//===----------------------------------------------------------------------===//
// Trait classes for Profile information.
//===----------------------------------------------------------------------===//

/// Generic profile template.  The default behavior is to invoke the
/// profile method of an object.  Specializations for primitive integers
/// and generic handling of pointers is done below.
template <typename T>
struct ImutProfileInfo {
  using value_type = const T;
  using value_type_ref = const T&;

  static void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    FoldingSetTrait<T>::Profile(X,ID);
  }
};

/// Profile traits for integers.
template <typename T>
struct ImutProfileInteger {
  using value_type = const T;
  using value_type_ref = const T&;

  static void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    ID.AddInteger(X);
  }
};

#define PROFILE_INTEGER_INFO(X)\
template<> struct ImutProfileInfo<X> : ImutProfileInteger<X> {};

PROFILE_INTEGER_INFO(char)
PROFILE_INTEGER_INFO(unsigned char)
PROFILE_INTEGER_INFO(short)
PROFILE_INTEGER_INFO(unsigned short)
PROFILE_INTEGER_INFO(unsigned)
PROFILE_INTEGER_INFO(signed)
PROFILE_INTEGER_INFO(long)
PROFILE_INTEGER_INFO(unsigned long)
PROFILE_INTEGER_INFO(long long)
PROFILE_INTEGER_INFO(unsigned long long)

#undef PROFILE_INTEGER_INFO

/// Profile traits for booleans.
template <>
struct ImutProfileInfo<bool> {
  using value_type = const bool;
  using value_type_ref = const bool&;

  static void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    ID.AddBoolean(X);
  }
};

/// Generic profile trait for pointer types.  We treat pointers as
/// references to unique objects.
template <typename T>
struct ImutProfileInfo<T*> {
  using value_type = const T*;
  using value_type_ref = value_type;

  static void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    ID.AddPointer(X);
  }
};

//===----------------------------------------------------------------------===//
// Trait classes that contain element comparison operators and type
//  definitions used by ImutAVLTree, ImmutableSet, and ImmutableMap.  These
//  inherit from the profile traits (ImutProfileInfo) to include operations
//  for element profiling.
//===----------------------------------------------------------------------===//

/// Generic definition of comparison operations for elements of immutable
/// containers that defaults to using std::equal_to<> and std::less<> to perform
/// comparison of elements.
template <typename T> struct ImutContainerInfo : ImutProfileInfo<T> {
  using value_type = typename ImutProfileInfo<T>::value_type;
  using value_type_ref = typename ImutProfileInfo<T>::value_type_ref;
  using key_type = value_type;
  using key_type_ref = value_type_ref;
  using data_type = bool;
  using data_type_ref = bool;

  static key_type_ref KeyOfValue(value_type_ref D) { return D; }
  static data_type_ref DataOfValue(value_type_ref) { return true; }

  static bool isEqual(key_type_ref LHS, key_type_ref RHS) {
    return std::equal_to<key_type>()(LHS,RHS);
  }

  static bool isLess(key_type_ref LHS, key_type_ref RHS) {
    return std::less<key_type>()(LHS,RHS);
  }

  static bool isDataEqual(data_type_ref, data_type_ref) { return true; }
};

/// Specialization for pointer values to treat pointers as references to unique
/// objects. Pointers are thus compared by their addresses.
template <typename T> struct ImutContainerInfo<T *> : ImutProfileInfo<T *> {
  using value_type = typename ImutProfileInfo<T*>::value_type;
  using value_type_ref = typename ImutProfileInfo<T*>::value_type_ref;
  using key_type = value_type;
  using key_type_ref = value_type_ref;
  using data_type = bool;
  using data_type_ref = bool;

  static key_type_ref KeyOfValue(value_type_ref D) { return D; }
  static data_type_ref DataOfValue(value_type_ref) { return true; }

  static bool isEqual(key_type_ref LHS, key_type_ref RHS) { return LHS == RHS; }

  static bool isLess(key_type_ref LHS, key_type_ref RHS) { return LHS < RHS; }

  static bool isDataEqual(data_type_ref, data_type_ref) { return true; }
};

//===----------------------------------------------------------------------===//
// Immutable Set
//===----------------------------------------------------------------------===//

template <typename ValT, typename ValInfo = ImutContainerInfo<ValT>,
          bool Canonicalize = true>
class ImmutableSet {
public:
  using value_type = typename ValInfo::value_type;
  using value_type_ref = typename ValInfo::value_type_ref;
  using TreeTy = ImutAVLTree<ValInfo, Canonicalize>;

private:
  IntrusiveRefCntPtr<TreeTy> Root;

public:
  /// Constructs a set from a pointer to a tree root.  In general one
  /// should use a Factory object to create sets instead of directly
  /// invoking the constructor, but there are cases where make this
  /// constructor public is useful.
  explicit ImmutableSet(TreeTy *R) : Root(R) {}

  class Factory {
    typename TreeTy::Factory F;

  public:
    Factory() = default;

    Factory(BumpPtrAllocator &Alloc) : F(Alloc) {}

    Factory(const Factory& RHS) = delete;
    void operator=(const Factory& RHS) = delete;

    /// Returns an immutable set that contains no elements.
    ImmutableSet getEmptySet() {
      return ImmutableSet(F.getEmptyTree());
    }

    /// Creates a new immutable set that contains all of the values
    /// of the original set with the addition of the specified value.  If
    /// the original set already included the value, then the original set is
    /// returned and no memory is allocated.  The time and space complexity
    /// of this operation is logarithmic in the size of the original set.
    /// The memory allocated to represent the set is released when the
    /// factory object that created the set is destroyed.
    [[nodiscard]] ImmutableSet add(ImmutableSet Old, value_type_ref V) {
      TreeTy *NewT = F.add(Old.Root.get(), V);
      if constexpr (Canonicalize)
        return ImmutableSet(F.getCanonicalTree(NewT));
      else
        return ImmutableSet(NewT);
    }

    /// Returns the union of \p A and \p B, computed in a single traversal that
    /// shares subtrees of both operands wherever possible (see
    /// ImutAVLFactory::unionTrees). This is more efficient than repeatedly
    /// adding \p B's elements to \p A when \p B is large.
    [[nodiscard]] ImmutableSet unionSets(ImmutableSet A, ImmutableSet B) {
      if (A.Root.get() == B.Root.get() || B.isEmpty())
        return A;
      if (A.isEmpty())
        return B;
      // Drive the recursion with the taller tree so the shorter one is the one
      // being split.
      if (A.getHeight() < B.getHeight())
        std::swap(A, B);
      if constexpr (Canonicalize) {
        // The bulk path does not canonicalize the nodes it creates, so fall
        // back to per-element insertion for canonicalizing factories.
        for (value_type_ref V : B)
          A = add(A, V);
        return A;
      } else {
        return ImmutableSet(F.unionTrees(A.Root.get(), B.Root.get()));
      }
    }

    /// Creates a new immutable set that contains all of the values
    /// of the original set with the exception of the specified value.  If
    /// the original set did not contain the value, the original set is
    /// returned and no memory is allocated.  The time and space complexity
    /// of this operation is logarithmic in the size of the original set.
    /// The memory allocated to represent the set is released when the
    /// factory object that created the set is destroyed.
    [[nodiscard]] ImmutableSet remove(ImmutableSet Old, value_type_ref V) {
      TreeTy *NewT = F.remove(Old.Root.get(), V);
      if constexpr (Canonicalize)
        return ImmutableSet(F.getCanonicalTree(NewT));
      else
        return ImmutableSet(NewT);
    }

    BumpPtrAllocator& getAllocator() { return F.getAllocator(); }

    typename TreeTy::Factory *getTreeFactory() const {
      return const_cast<typename TreeTy::Factory *>(&F);
    }
  };

  friend class Factory;

  /// Returns true if the set contains the specified value.
  bool contains(value_type_ref V) const {
    return Root ? Root->contains(V) : false;
  }

  /// Compares two sets for equality. For a canonicalizing factory, sets with
  /// equal contents share the same tree, so this is an O(1) pointer comparison
  /// (like ImmutableList); only sets created by the same factory may be
  /// compared. Otherwise it is a structural comparison.
  bool operator==(const ImmutableSet &RHS) const {
    if constexpr (Canonicalize)
      return Root == RHS.Root;
    else
      return Root && RHS.Root ? Root->isEqual(*RHS.Root.get())
                              : Root == RHS.Root;
  }

  bool operator!=(const ImmutableSet &RHS) const {
    if constexpr (Canonicalize)
      return Root != RHS.Root;
    else
      return Root && RHS.Root ? Root->isNotEqual(*RHS.Root.get())
                              : Root != RHS.Root;
  }

  TreeTy *getRoot() {
    if (Root) { Root->retain(); }
    return Root.get();
  }

  TreeTy *getRootWithoutRetain() const { return Root.get(); }

  /// Return true if the set contains no elements.
  bool isEmpty() const { return !Root; }

  /// Return true if the set contains exactly one element.
  /// This method runs in constant time.
  bool isSingleton() const { return getHeight() == 1; }

  //===--------------------------------------------------===//
  // Iterators.
  //===--------------------------------------------------===//

  using iterator = ImutAVLValueIterator<ImmutableSet>;

  iterator begin() const { return iterator(Root.get()); }
  iterator end() const { return iterator(); }

  //===--------------------------------------------------===//
  // Utility methods.
  //===--------------------------------------------------===//

  unsigned getHeight() const { return Root ? Root->getHeight() : 0; }

  static void Profile(FoldingSetNodeID &ID, const ImmutableSet &S) {
    ID.AddPointer(S.Root.get());
  }

  void Profile(FoldingSetNodeID &ID) const { return Profile(ID, *this); }

  //===--------------------------------------------------===//
  // For testing.
  //===--------------------------------------------------===//

  void validateTree() const { if (Root) Root->validateTree(); }
};

// NOTE: This may some day replace the current ImmutableSet.
template <typename ValT, typename ValInfo = ImutContainerInfo<ValT>,
          bool Canonicalize = true>
class ImmutableSetRef {
public:
  using value_type = typename ValInfo::value_type;
  using value_type_ref = typename ValInfo::value_type_ref;
  using TreeTy = ImutAVLTree<ValInfo, Canonicalize>;
  using FactoryTy = typename TreeTy::Factory;

private:
  IntrusiveRefCntPtr<TreeTy> Root;
  FactoryTy *Factory;

public:
  /// Constructs a set from a pointer to a tree root.  In general one
  /// should use a Factory object to create sets instead of directly
  /// invoking the constructor, but there are cases where make this
  /// constructor public is useful.
  ImmutableSetRef(TreeTy *R, FactoryTy *F) : Root(R), Factory(F) {}

  static ImmutableSetRef getEmptySet(FactoryTy *F) {
    return ImmutableSetRef(0, F);
  }

  ImmutableSetRef add(value_type_ref V) {
    return ImmutableSetRef(Factory->add(Root.get(), V), Factory);
  }

  ImmutableSetRef remove(value_type_ref V) {
    return ImmutableSetRef(Factory->remove(Root.get(), V), Factory);
  }

  /// Returns true if the set contains the specified value.
  bool contains(value_type_ref V) const {
    return Root ? Root->contains(V) : false;
  }

  ImmutableSet<ValT, ValInfo, Canonicalize> asImmutableSet() const {
    using SetTy = ImmutableSet<ValT, ValInfo, Canonicalize>;
    if constexpr (Canonicalize)
      return SetTy(Factory->getCanonicalTree(Root.get()));
    else
      return SetTy(Root.get());
  }

  TreeTy *getRootWithoutRetain() const { return Root.get(); }

  bool operator==(const ImmutableSetRef &RHS) const {
    return Root && RHS.Root ? Root->isEqual(*RHS.Root.get()) : Root == RHS.Root;
  }

  bool operator!=(const ImmutableSetRef &RHS) const {
    return Root && RHS.Root ? Root->isNotEqual(*RHS.Root.get())
                            : Root != RHS.Root;
  }

  /// Return true if the set contains no elements.
  bool isEmpty() const { return !Root; }

  /// Return true if the set contains exactly one element.
  /// This method runs in constant time.
  bool isSingleton() const { return getHeight() == 1; }

  //===--------------------------------------------------===//
  // Iterators.
  //===--------------------------------------------------===//

  using iterator = ImutAVLValueIterator<ImmutableSetRef>;

  iterator begin() const { return iterator(Root.get()); }
  iterator end() const { return iterator(); }

  //===--------------------------------------------------===//
  // Utility methods.
  //===--------------------------------------------------===//

  unsigned getHeight() const { return Root ? Root->getHeight() : 0; }

  static void Profile(FoldingSetNodeID &ID, const ImmutableSetRef &S) {
    ID.AddPointer(S.Root.get());
  }

  void Profile(FoldingSetNodeID &ID) const { return Profile(ID, *this); }

  //===--------------------------------------------------===//
  // For testing.
  //===--------------------------------------------------===//

  void validateTree() const { if (Root) Root->validateTree(); }
};

} // end namespace llvm

#endif // LLVM_ADT_IMMUTABLESET_H
