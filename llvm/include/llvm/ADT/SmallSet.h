//===- llvm/ADT/SmallSet.h - 'Normally small' sets --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallSet class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLSET_H
#define LLVM_ADT_SMALLSET_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>

namespace llvm {

/// SmallSet - This maintains a set of unique values, optimizing for the case
/// when the set is small (less than N).  In this case, the set can be
/// maintained with no mallocs.  If the set gets large, we expand to using an
/// std::set to maintain reasonable lookup times.
///
/// Note that this set does not provide a way to iterate over members in the
/// set.
template <typename T, unsigned N,  typename C = std::less<T> >
class SmallSet {
  /// Use a SmallVector to hold the elements here (even though it will never
  /// reach its 'large' stage) to avoid calling the default ctors of elements
  /// we will never use.
  SmallVector<T, N> Vector;
  std::set<T, C> Set;
  typedef typename SmallVector<T, N>::const_iterator VIterator;
  typedef typename SmallVector<T, N>::iterator mutable_iterator;
public:
  SmallSet() {}

  bool empty() const { return Vector.empty() && Set.empty(); }
  unsigned size() const {
    return isSmall() ? Vector.size() : Set.size();
  }

  /// count - Return true if the element is in the set.
  bool count(const T &V) const {
    if (isSmall()) {
      // Since the collection is small, just do a linear search.
      return vfind(V) != Vector.end();
    } else {
      return Set.count(V);
    }
  }

  /// insert - Insert an element into the set if it isn't already there.
  bool insert(const T &V) {
    if (!isSmall())
      return Set.insert(V).second;

    VIterator I = vfind(V);
    if (I != Vector.end())    // Don't reinsert if it already exists.
      return false;
    if (Vector.size() < N) {
      Vector.push_back(V);
      return true;
    }

    // Otherwise, grow from vector to set.
    while (!Vector.empty()) {
      Set.insert(Vector.back());
      Vector.pop_back();
    }
    Set.insert(V);
    return true;
  }

  template <typename IterT>
  void insert(IterT I, IterT E) {
    for (; I != E; ++I)
      insert(*I);
  }
  
  bool erase(const T &V) {
    if (!isSmall())
      return Set.erase(V);
    for (mutable_iterator I = Vector.begin(), E = Vector.end(); I != E; ++I)
      if (*I == V) {
        Vector.erase(I);
        return true;
      }
    return false;
  }

  void clear() {
    Vector.clear();
    Set.clear();
  }
private:
  bool isSmall() const { return Set.empty(); }

  VIterator vfind(const T &V) const {
    for (VIterator I = Vector.begin(), E = Vector.end(); I != E; ++I)
      if (*I == V)
        return I;
    return Vector.end();
  }
};

/// If this set is of pointer values, transparently switch over to using
/// SmallPtrSet for performance.
template <typename PointeeType, unsigned N>
class SmallSet<PointeeType*, N> : public SmallPtrSet<PointeeType*, N> {};

} // end namespace llvm

#endif
