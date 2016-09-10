//===- llvm/ADT/simple_ilist.h - Simple Intrusive List ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SIMPLE_ILIST_H
#define LLVM_ADT_SIMPLE_ILIST_H

#include "llvm/ADT/ilist_base.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/ilist_node.h"
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace llvm {

/// A simple intrusive list implementation.
///
/// This is a simple intrusive list for a \c T that inherits from \c
/// ilist_node<T>.  The list never takes ownership of anything inserted in it.
///
/// Unlike \a iplist<T> and \a ilist<T>, \a simple_ilist<T> never allocates or
/// deletes values, and has no callback traits.
///
/// The API for adding nodes include \a push_front(), \a push_back(), and \a
/// insert().  These all take values by reference (not by pointer), except for
/// the range version of \a insert().
///
/// There are three sets of API for discarding nodes from the list: \a
/// remove(), which takes a reference to the node to remove, \a erase(), which
/// takes an iterator or iterator range and returns the next one, and \a
/// clear(), which empties out the container.  All three are constant time
/// operations.  None of these deletes any nodes; in particular, if there is a
/// single node in the list, then these have identical semantics:
/// \li \c L.remove(L.front());
/// \li \c L.erase(L.begin());
/// \li \c L.clear();
///
/// As a convenience for callers, there are parallel APIs that take a \c
/// Disposer (such as \c std::default_delete<T>): \a removeAndDispose(), \a
/// eraseAndDispose(), and \a clearAndDispose().  These have different names
/// because the extra semantic is otherwise non-obvious.  They are equivalent
/// to calling \a std::for_each() on the range to be discarded.
template <typename T>
class simple_ilist : ilist_base, ilist_detail::SpecificNodeAccess<T> {
  typedef ilist_base list_base_type;
  ilist_sentinel<T> Sentinel;

public:
  typedef T value_type;
  typedef T *pointer;
  typedef T &reference;
  typedef const T *const_pointer;
  typedef const T &const_reference;
  typedef ilist_iterator<T> iterator;
  typedef ilist_iterator<const T> const_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef ilist_iterator<const T, true> const_reverse_iterator;
  typedef ilist_iterator<T, true> reverse_iterator;

  simple_ilist() = default;
  ~simple_ilist() = default;

  // No copy constructors.
  simple_ilist(const simple_ilist &) = delete;
  simple_ilist &operator=(const simple_ilist &) = delete;

  // Move constructors.
  simple_ilist(simple_ilist &&X) { splice(end(), X); }
  simple_ilist &operator=(simple_ilist &&X) {
    clear();
    splice(end(), X);
    return *this;
  }

  iterator begin() { return ++iterator(Sentinel); }
  const_iterator begin() const { return ++const_iterator(Sentinel); }
  iterator end() { return iterator(Sentinel); }
  const_iterator end() const { return const_iterator(Sentinel); }
  reverse_iterator rbegin() { return ++reverse_iterator(Sentinel); }
  const_reverse_iterator rbegin() const {
    return ++const_reverse_iterator(Sentinel);
  }
  reverse_iterator rend() { return reverse_iterator(Sentinel); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(Sentinel);
  }

  /// Check if the list is empty in constant time.
  bool LLVM_ATTRIBUTE_UNUSED_RESULT empty() const { return Sentinel.empty(); }

  /// Calculate the size of the list in linear time.
  size_type LLVM_ATTRIBUTE_UNUSED_RESULT size() const {
    return std::distance(begin(), end());
  }

  reference front() { return *begin(); }
  const_reference front() const { return *begin(); }
  reference back() { return *rbegin(); }
  const_reference back() const { return *rbegin(); }

  /// Insert a node at the front; never copies.
  void push_front(reference Node) { insert(begin(), Node); }

  /// Insert a node at the back; never copies.
  void push_back(reference Node) { insert(end(), Node); }

  /// Remove the node at the front; never deletes.
  void pop_front() { erase(begin()); }

  /// Remove the node at the back; never deletes.
  void pop_back() { erase(--end()); }

  /// Swap with another list in place using std::swap.
  void swap(simple_ilist &X) { std::swap(*this, X); }

  /// Insert a node by reference; never copies.
  iterator insert(iterator I, reference Node) {
    list_base_type::insertBefore(*I.getNodePtr(), *this->getNodePtr(&Node));
    return iterator(&Node);
  }

  /// Insert a range of nodes; never copies.
  template <class Iterator>
  void insert(iterator I, Iterator First, Iterator Last) {
    for (; First != Last; ++First)
      insert(I, *First);
  }

  /// Remove a node by reference; never deletes.
  ///
  /// \see \a erase() for removing by iterator.
  /// \see \a removeAndDispose() if the node should be deleted.
  void remove(reference N) { list_base_type::remove(*this->getNodePtr(&N)); }

  /// Remove a node by reference and dispose of it.
  template <class Disposer>
  void removeAndDispose(reference N, Disposer dispose) {
    remove(N);
    dispose(&N);
  }

  /// Remove a node by iterator; never deletes.
  ///
  /// \see \a remove() for removing by reference.
  /// \see \a eraseAndDispose() it the node should be deleted.
  iterator erase(iterator I) {
    assert(I != end() && "Cannot remove end of list!");
    remove(*I++);
    return I;
  }

  /// Remove a range of nodes; never deletes.
  ///
  /// \see \a eraseAndDispose() if the nodes should be deleted.
  iterator erase(iterator First, iterator Last) {
    list_base_type::removeRange(*First.getNodePtr(), *Last.getNodePtr());
    return Last;
  }

  /// Remove a node by iterator and dispose of it.
  template <class Disposer>
  iterator eraseAndDispose(iterator I, Disposer dispose) {
    auto Next = std::next(I);
    erase(I);
    dispose(&*I);
    return Next;
  }

  /// Remove a range of nodes and dispose of them.
  template <class Disposer>
  iterator eraseAndDispose(iterator First, iterator Last, Disposer dispose) {
    while (First != Last)
      First = eraseAndDispose(First, dispose);
    return Last;
  }

  /// Clear the list; never deletes.
  ///
  /// \see \a clearAndDispose() if the nodes should be deleted.
  void clear() { Sentinel.reset(); }

  /// Clear the list and dispose of the nodes.
  template <class Disposer> void clearAndDispose(Disposer dispose) {
    eraseAndDispose(begin(), end(), dispose);
  }

  /// Splice in another list.
  void splice(iterator I, simple_ilist &L2) {
    splice(I, L2, L2.begin(), L2.end());
  }

  /// Splice in a node from another list.
  void splice(iterator I, simple_ilist &L2, iterator Node) {
    splice(I, L2, Node, std::next(Node));
  }

  /// Splice in a range of nodes from another list.
  void splice(iterator I, simple_ilist &, iterator First, iterator Last) {
    list_base_type::transferBefore(*I.getNodePtr(), *First.getNodePtr(),
                                   *Last.getNodePtr());
  }

  /// Merge in another list.
  ///
  /// \pre \c this and \p RHS are sorted.
  ///@{
  void merge(simple_ilist &RHS) { merge(RHS, std::less<T>()); }
  template <class Compare> void merge(simple_ilist &RHS, Compare comp);
  ///@}

  /// Sort the list.
  ///@{
  void sort() { sort(std::less<T>()); }
  template <class Compare> void sort(Compare comp);
  ///@}
};

template <class T>
template <class Compare>
void simple_ilist<T>::merge(simple_ilist<T> &RHS, Compare comp) {
  if (this == &RHS || RHS.empty())
    return;
  iterator LI = begin(), LE = end();
  iterator RI = RHS.begin(), RE = RHS.end();
  while (LI != LE) {
    if (comp(*RI, *LI)) {
      // Transfer a run of at least size 1 from RHS to LHS.
      iterator RunStart = RI++;
      RI = std::find_if(RI, RE, [&](reference RV) { return !comp(RV, *LI); });
      splice(LI, RHS, RunStart, RI);
      if (RI == RE)
        return;
    }
    ++LI;
  }
  // Transfer the remaining RHS nodes once LHS is finished.
  splice(LE, RHS, RI, RE);
}

template <class T>
template <class Compare>
void simple_ilist<T>::sort(Compare comp) {
  // Vacuously sorted.
  if (empty() || std::next(begin()) == end())
    return;

  // Split the list in the middle.
  iterator Center = begin(), End = begin();
  while (End != end() && ++End != end()) {
    ++Center;
    ++End;
  }
  simple_ilist<T> RHS;
  RHS.splice(RHS.end(), *this, Center, end());

  // Sort the sublists and merge back together.
  sort(comp);
  RHS.sort(comp);
  merge(RHS, comp);
}

} // end namespace llvm

#endif // LLVM_ADT_SIMPLE_ILIST_H
