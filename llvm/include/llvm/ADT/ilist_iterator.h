//===- llvm/ADT/ilist_iterator.h - Intrusive List Iterator -------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_ITERATOR_H
#define LLVM_ADT_ILIST_ITERATOR_H

#include "llvm/ADT/ilist_node.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace llvm {

namespace ilist_detail {

template <class NodeTy> struct ConstCorrectNodeType {
  typedef ilist_node<NodeTy> type;
};
template <class NodeTy> struct ConstCorrectNodeType<const NodeTy> {
  typedef const ilist_node<NodeTy> type;
};

template <bool IsReverse = false> struct IteratorHelper {
  template <class T> static void increment(T *&I) {
    I = ilist_node_access::getNext(*I);
  }
  template <class T> static void decrement(T *&I) {
    I = ilist_node_access::getPrev(*I);
  }
};
template <> struct IteratorHelper<true> {
  template <class T> static void increment(T *&I) {
    IteratorHelper<false>::decrement(I);
  }
  template <class T> static void decrement(T *&I) {
    IteratorHelper<false>::increment(I);
  }
};

} // end namespace ilist_detail

/// Iterator for intrusive lists  based on ilist_node.
template <typename NodeTy, bool IsReverse> class ilist_iterator {
public:
  typedef NodeTy value_type;
  typedef value_type *pointer;
  typedef value_type &reference;
  typedef ptrdiff_t difference_type;
  typedef std::bidirectional_iterator_tag iterator_category;

  typedef typename std::add_const<value_type>::type *const_pointer;
  typedef typename std::add_const<value_type>::type &const_reference;

  typedef typename ilist_detail::ConstCorrectNodeType<NodeTy>::type node_type;
  typedef node_type *node_pointer;
  typedef node_type &node_reference;

private:
  node_pointer NodePtr;

public:
  /// Create from an ilist_node.
  explicit ilist_iterator(node_reference N) : NodePtr(&N) {}

  explicit ilist_iterator(pointer NP)
      : NodePtr(ilist_node_access::getNodePtr(NP)) {}
  explicit ilist_iterator(reference NR)
      : NodePtr(ilist_node_access::getNodePtr(&NR)) {}
  ilist_iterator() : NodePtr(nullptr) {}

  // This is templated so that we can allow constructing a const iterator from
  // a nonconst iterator...
  template <class node_ty>
  ilist_iterator(
      const ilist_iterator<node_ty, IsReverse> &RHS,
      typename std::enable_if<std::is_convertible<node_ty *, NodeTy *>::value,
                              void *>::type = nullptr)
      : NodePtr(RHS.getNodePtr()) {}

  // This is templated so that we can allow assigning to a const iterator from
  // a nonconst iterator...
  template <class node_ty>
  const ilist_iterator &
  operator=(const ilist_iterator<node_ty, IsReverse> &RHS) {
    NodePtr = RHS.getNodePtr();
    return *this;
  }

  /// Convert from an iterator to its reverse.
  ///
  /// TODO: Roll this into the implicit constructor once we're sure that no one
  /// is relying on the std::reverse_iterator off-by-one semantics.
  ilist_iterator<NodeTy, !IsReverse> getReverse() const {
    if (NodePtr)
      return ilist_iterator<NodeTy, !IsReverse>(*NodePtr);
    return ilist_iterator<NodeTy, !IsReverse>();
  }

  void reset(pointer NP) { NodePtr = NP; }

  // Accessors...
  reference operator*() const {
    assert(!NodePtr->isKnownSentinel());
    return *ilist_node_access::getValuePtr(NodePtr);
  }
  pointer operator->() const { return &operator*(); }

  // Comparison operators
  friend bool operator==(const ilist_iterator &LHS, const ilist_iterator &RHS) {
    return LHS.NodePtr == RHS.NodePtr;
  }
  friend bool operator!=(const ilist_iterator &LHS, const ilist_iterator &RHS) {
    return LHS.NodePtr != RHS.NodePtr;
  }

  // Increment and decrement operators...
  ilist_iterator &operator--() {
    ilist_detail::IteratorHelper<IsReverse>::decrement(NodePtr);
    return *this;
  }
  ilist_iterator &operator++() {
    ilist_detail::IteratorHelper<IsReverse>::increment(NodePtr);
    return *this;
  }
  ilist_iterator operator--(int) {
    ilist_iterator tmp = *this;
    --*this;
    return tmp;
  }
  ilist_iterator operator++(int) {
    ilist_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  /// Get the underlying ilist_node.
  node_pointer getNodePtr() const { return static_cast<node_pointer>(NodePtr); }
};

template <typename From> struct simplify_type;

/// Allow ilist_iterators to convert into pointers to a node automatically when
/// used by the dyn_cast, cast, isa mechanisms...
///
/// FIXME: remove this, since there is no implicit conversion to NodeTy.
template <typename NodeTy> struct simplify_type<ilist_iterator<NodeTy>> {
  typedef NodeTy *SimpleType;

  static SimpleType getSimplifiedValue(ilist_iterator<NodeTy> &Node) {
    return &*Node;
  }
};
template <typename NodeTy> struct simplify_type<const ilist_iterator<NodeTy>> {
  typedef /*const*/ NodeTy *SimpleType;

  static SimpleType getSimplifiedValue(const ilist_iterator<NodeTy> &Node) {
    return &*Node;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_ILIST_ITERATOR_H
