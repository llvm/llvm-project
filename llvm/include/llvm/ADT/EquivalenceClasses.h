//===- llvm/ADT/EquivalenceClasses.h - Generic Equiv. Classes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generic implementation of equivalence classes through the use Tarjan's
/// efficient union-find algorithm.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_EQUIVALENCECLASSES_H
#define LLVM_ADT_EQUIVALENCECLASSES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace llvm {

/// EquivalenceClasses - This represents a collection of equivalence classes and
/// supports three efficient operations: insert an element into a class of its
/// own, union two classes, and find the class for a given element.  In
/// addition to these modification methods, it is possible to iterate over all
/// of the equivalence classes and all of the elements in a class.
///
/// This implementation is an efficient implementation that only stores one copy
/// of the element being indexed per entry in the set, and allows any arbitrary
/// type to be indexed (as long as it can be implements DenseMapInfo).
///
/// Here is a simple example using integers:
///
/// \code
///  EquivalenceClasses<int> EC;
///  EC.unionSets(1, 2);                // insert 1, 2 into the same set
///  EC.insert(4); EC.insert(5);        // insert 4, 5 into own sets
///  EC.unionSets(5, 1);                // merge the set for 1 with 5's set.
///
///  for (EquivalenceClasses<int>::iterator I = EC.begin(), E = EC.end();
///       I != E; ++I) {           // Iterate over all of the equivalence sets.
///    if (!I->isLeader()) continue;   // Ignore non-leader sets.
///    for (EquivalenceClasses<int>::member_iterator MI = EC.member_begin(I);
///         MI != EC.member_end(); ++MI)   // Loop over members in this set.
///      cerr << *MI << " ";  // Print member.
///    cerr << "\n";   // Finish set.
///  }
/// \endcode
///
/// This example prints:
///   4
///   5 1 2
///
template <class ElemTy> class EquivalenceClasses {
public:
  /// ECValue - The EquivalenceClasses data structure is just a set of these.
  /// Each of these represents a relation for a value.  First it stores the
  /// value itself. Next, it provides a "next pointer", which is used to
  /// enumerate all of the elements in the unioned set.  Finally, it defines
  /// either a "end of list pointer" or "leader pointer" depending on whether
  /// the value itself is a leader. A "leader pointer" points to the node that
  /// is the leader for this element, if the node is not a leader.  A "end of
  /// list pointer" points to the last node in the list of members of this list.
  /// Whether or not a node is a leader is determined by a bit stolen from one
  /// of the pointers.
  class ECValue {
    friend class EquivalenceClasses;

    mutable const ECValue *Leader, *Next;
    ElemTy Data;

    // ECValue ctor - Start out with EndOfList pointing to this node, Next is
    // Null, isLeader = true.
    ECValue(const ElemTy &Elt)
        : Leader(this),
          Next(reinterpret_cast<ECValue *>(static_cast<intptr_t>(1))),
          Data(Elt) {}

    const ECValue *getLeader() const {
      if (isLeader())
        return this;
      if (Leader->isLeader())
        return Leader;
      // Path compression.
      return Leader = Leader->getLeader();
    }

    const ECValue *getEndOfList() const {
      assert(isLeader() && "Cannot get the end of a list for a non-leader!");
      return Leader;
    }

    void setNext(const ECValue *NewNext) const {
      assert(getNext() == nullptr && "Already has a next pointer!");
      Next = reinterpret_cast<const ECValue *>(
          reinterpret_cast<intptr_t>(NewNext) |
          static_cast<intptr_t>(isLeader()));
    }

  public:
    ECValue(const ECValue &RHS)
        : Leader(this),
          Next(reinterpret_cast<ECValue *>(static_cast<intptr_t>(1))),
          Data(RHS.Data) {
      // Only support copying of singleton nodes.
      assert(RHS.isLeader() && RHS.getNext() == nullptr && "Not a singleton!");
    }

    bool isLeader() const { return (intptr_t)Next & 1; }
    const ElemTy &getData() const { return Data; }

    const ECValue *getNext() const {
      return reinterpret_cast<ECValue *>(reinterpret_cast<intptr_t>(Next) &
                                         ~static_cast<intptr_t>(1));
    }
  };

private:
  /// TheMapping - This implicitly provides a mapping from ElemTy values to the
  /// ECValues, it just keeps the key as part of the value.
  DenseMap<ElemTy, ECValue *> TheMapping;

  /// List of all members, used to provide a deterministic iteration order.
  SmallVector<const ECValue *> Members;

  mutable BumpPtrAllocator ECValueAllocator;

public:
  EquivalenceClasses() = default;
  EquivalenceClasses(const EquivalenceClasses &RHS) { operator=(RHS); }

  EquivalenceClasses &operator=(const EquivalenceClasses &RHS) {
    TheMapping.clear();
    Members.clear();
    for (const auto &E : RHS)
      if (E->isLeader()) {
        member_iterator MI = RHS.member_begin(*E);
        member_iterator LeaderIt = member_begin(insert(*MI));
        for (++MI; MI != member_end(); ++MI)
          unionSets(LeaderIt, member_begin(insert(*MI)));
      }
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // Inspection methods
  //

  /// iterator* - Provides a way to iterate over all values in the set.
  using iterator = typename SmallVector<const ECValue *>::const_iterator;

  iterator begin() const { return Members.begin(); }
  iterator end() const { return Members.end(); }

  bool empty() const { return TheMapping.empty(); }

  /// member_* Iterate over the members of an equivalence class.
  class member_iterator;
  member_iterator member_begin(const ECValue &ECV) const {
    // Only leaders provide anything to iterate over.
    return member_iterator(ECV.isLeader() ? &ECV : nullptr);
  }

  member_iterator member_end() const { return member_iterator(nullptr); }

  iterator_range<member_iterator> members(const ECValue &ECV) const {
    return make_range(member_begin(ECV), member_end());
  }

  iterator_range<member_iterator> members(const ElemTy &V) const {
    return make_range(findLeader(V), member_end());
  }

  /// Returns true if \p V is contained an equivalence class.
  [[nodiscard]] bool contains(const ElemTy &V) const {
    return TheMapping.contains(V);
  }

  /// getLeaderValue - Return the leader for the specified value that is in the
  /// set.  It is an error to call this method for a value that is not yet in
  /// the set.  For that, call getOrInsertLeaderValue(V).
  const ElemTy &getLeaderValue(const ElemTy &V) const {
    member_iterator MI = findLeader(V);
    assert(MI != member_end() && "Value is not in the set!");
    return *MI;
  }

  /// getOrInsertLeaderValue - Return the leader for the specified value that is
  /// in the set.  If the member is not in the set, it is inserted, then
  /// returned.
  const ElemTy &getOrInsertLeaderValue(const ElemTy &V) {
    member_iterator MI = findLeader(insert(V));
    assert(MI != member_end() && "Value is not in the set!");
    return *MI;
  }

  /// getNumClasses - Return the number of equivalence classes in this set.
  /// Note that this is a linear time operation.
  unsigned getNumClasses() const {
    unsigned NC = 0;
    for (const auto &E : *this)
      if (E->isLeader())
        ++NC;
    return NC;
  }

  //===--------------------------------------------------------------------===//
  // Mutation methods

  /// insert - Insert a new value into the union/find set, ignoring the request
  /// if the value already exists.
  const ECValue &insert(const ElemTy &Data) {
    auto [I, Inserted] = TheMapping.try_emplace(Data);
    if (!Inserted)
      return *I->second;

    auto *ECV = new (ECValueAllocator) ECValue(Data);
    I->second = ECV;
    Members.push_back(ECV);
    return *ECV;
  }

  /// erase - Erase a value from the union/find set, return "true" if erase
  /// succeeded, or "false" when the value was not found.
  bool erase(const ElemTy &V) {
    if (!TheMapping.contains(V))
      return false;
    const ECValue *Cur = TheMapping[V];
    const ECValue *Next = Cur->getNext();
    // If the current element is the leader and has a successor element,
    // update the successor element's 'Leader' field to be the last element,
    // set the successor element's stolen bit, and set the 'Leader' field of
    // all other elements in same class to be the successor element.
    if (Cur->isLeader() && Next) {
      Next->Leader = Cur->Leader;
      Next->Next = reinterpret_cast<const ECValue *>(
          reinterpret_cast<intptr_t>(Next->Next) | static_cast<intptr_t>(1));

      const ECValue *NewLeader = Next;
      while ((Next = Next->getNext())) {
        Next->Leader = NewLeader;
      }
    } else if (!Cur->isLeader()) {
      const ECValue *Leader = findLeader(V).Node;
      const ECValue *Pre = Leader;
      while (Pre->getNext() != Cur) {
        Pre = Pre->getNext();
      }
      if (!Next) {
        // If the current element is the last element(not leader), set the
        // successor of the current element's predecessor to null while
        // preserving the leader bit, and set the 'Leader' field of the class
        // leader to the predecessor element.
        Pre->Next = reinterpret_cast<const ECValue *>(
            static_cast<intptr_t>(Pre->isLeader()));
        Leader->Leader = Pre;
      } else {
        // If the current element is in the middle of class, then simply
        // connect the predecessor element and the successor element.
        Pre->Next = reinterpret_cast<const ECValue *>(
            reinterpret_cast<intptr_t>(Next) |
            static_cast<intptr_t>(Pre->isLeader()));
        Next->Leader = Pre;
      }
    }

    // Update 'TheMapping' and 'Members'.
    assert(TheMapping.contains(V) && "Can't find input in TheMapping!");
    TheMapping.erase(V);
    auto I = find(Members, Cur);
    assert(I != Members.end() && "Can't find input in members!");
    Members.erase(I);
    return true;
  }

  /// findLeader - Given a value in the set, return a member iterator for the
  /// equivalence class it is in.  This does the path-compression part that
  /// makes union-find "union findy".  This returns an end iterator if the value
  /// is not in the equivalence class.
  member_iterator findLeader(const ElemTy &V) const {
    auto I = TheMapping.find(V);
    if (I == TheMapping.end())
      return member_iterator(nullptr);
    return findLeader(*I->second);
  }
  member_iterator findLeader(const ECValue &ECV) const {
    return member_iterator(ECV.getLeader());
  }

  /// union - Merge the two equivalence sets for the specified values, inserting
  /// them if they do not already exist in the equivalence set.
  member_iterator unionSets(const ElemTy &V1, const ElemTy &V2) {
    const ECValue &V1I = insert(V1), &V2I = insert(V2);
    return unionSets(findLeader(V1I), findLeader(V2I));
  }
  member_iterator unionSets(member_iterator L1, member_iterator L2) {
    assert(L1 != member_end() && L2 != member_end() && "Illegal inputs!");
    if (L1 == L2)
      return L1; // Unifying the same two sets, noop.

    // Otherwise, this is a real union operation.  Set the end of the L1 list to
    // point to the L2 leader node.
    const ECValue &L1LV = *L1.Node, &L2LV = *L2.Node;
    L1LV.getEndOfList()->setNext(&L2LV);

    // Update L1LV's end of list pointer.
    L1LV.Leader = L2LV.getEndOfList();

    // Clear L2's leader flag:
    L2LV.Next = L2LV.getNext();

    // L2's leader is now L1.
    L2LV.Leader = &L1LV;
    return L1;
  }

  // isEquivalent - Return true if V1 is equivalent to V2. This can happen if
  // V1 is equal to V2 or if they belong to one equivalence class.
  bool isEquivalent(const ElemTy &V1, const ElemTy &V2) const {
    // Fast path: any element is equivalent to itself.
    if (V1 == V2)
      return true;
    auto It = findLeader(V1);
    return It != member_end() && It == findLeader(V2);
  }

  class member_iterator {
    friend class EquivalenceClasses;

    const ECValue *Node;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const ElemTy;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    explicit member_iterator() = default;
    explicit member_iterator(const ECValue *N) : Node(N) {}

    reference operator*() const {
      assert(Node != nullptr && "Dereferencing end()!");
      return Node->getData();
    }
    pointer operator->() const { return &operator*(); }

    member_iterator &operator++() {
      assert(Node != nullptr && "++'d off the end of the list!");
      Node = Node->getNext();
      return *this;
    }

    member_iterator operator++(int) { // postincrement operators.
      member_iterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const member_iterator &RHS) const {
      return Node == RHS.Node;
    }
    bool operator!=(const member_iterator &RHS) const {
      return Node != RHS.Node;
    }
  };
};

} // end namespace llvm

#endif // LLVM_ADT_EQUIVALENCECLASSES_H
