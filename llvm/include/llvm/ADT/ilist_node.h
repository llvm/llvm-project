//==-- llvm/ADT/ilist_node.h - Intrusive Linked List Helper ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ilist_node class template, which is a convenient
// base class for creating classes that can be used with ilists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_NODE_H
#define LLVM_ADT_ILIST_NODE_H

#include "llvm/ADT/ilist_node_base.h"

namespace llvm {

namespace ilist_detail {
struct NodeAccess;
} // end namespace ilist_detail

template<typename NodeTy>
struct ilist_traits;

template <typename NodeTy, bool IsReverse = false> class ilist_iterator;
template <typename NodeTy> class ilist_sentinel;

/// Templated wrapper class.
template <typename NodeTy> class ilist_node : ilist_node_base {
  friend class ilist_base;
  friend struct ilist_detail::NodeAccess;
  friend struct ilist_traits<NodeTy>;
  friend class ilist_iterator<NodeTy, false>;
  friend class ilist_iterator<NodeTy, true>;
  friend class ilist_sentinel<NodeTy>;

protected:
  ilist_node() = default;

private:
  ilist_node *getPrev() {
    return static_cast<ilist_node *>(ilist_node_base::getPrev());
  }
  ilist_node *getNext() {
    return static_cast<ilist_node *>(ilist_node_base::getNext());
  }

  const ilist_node *getPrev() const {
    return static_cast<ilist_node *>(ilist_node_base::getPrev());
  }
  const ilist_node *getNext() const {
    return static_cast<ilist_node *>(ilist_node_base::getNext());
  }

  void setPrev(ilist_node *N) { ilist_node_base::setPrev(N); }
  void setNext(ilist_node *N) { ilist_node_base::setNext(N); }

public:
  ilist_iterator<NodeTy> getIterator() { return ilist_iterator<NodeTy>(*this); }
  ilist_iterator<const NodeTy> getIterator() const {
    return ilist_iterator<const NodeTy>(*this);
  }

  using ilist_node_base::isKnownSentinel;
};

namespace ilist_detail {
/// An access class for ilist_node private API.
///
/// This gives access to the private parts of ilist nodes.  Nodes for an ilist
/// should friend this class if they inherit privately from ilist_node.
///
/// Using this class outside of the ilist implementation is unsupported.
struct NodeAccess {
protected:
  template <typename T> static ilist_node<T> *getNodePtr(T *N) { return N; }
  template <typename T> static const ilist_node<T> *getNodePtr(const T *N) {
    return N;
  }
  template <typename T> static T *getValuePtr(ilist_node<T> *N) {
    return static_cast<T *>(N);
  }
  template <typename T> static const T *getValuePtr(const ilist_node<T> *N) {
    return static_cast<const T *>(N);
  }

  template <typename T> static ilist_node<T> *getPrev(ilist_node<T> &N) {
    return N.getPrev();
  }
  template <typename T> static ilist_node<T> *getNext(ilist_node<T> &N) {
    return N.getNext();
  }
  template <typename T>
  static const ilist_node<T> *getPrev(const ilist_node<T> &N) {
    return N.getPrev();
  }
  template <typename T>
  static const ilist_node<T> *getNext(const ilist_node<T> &N) {
    return N.getNext();
  }
};

template <class T> struct SpecificNodeAccess : NodeAccess {
protected:
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef ilist_node<T> node_type;

  static node_type *getNodePtr(pointer N) {
    return NodeAccess::getNodePtr<T>(N);
  }
  static const ilist_node<T> *getNodePtr(const_pointer N) {
    return NodeAccess::getNodePtr<T>(N);
  }
  static pointer getValuePtr(node_type *N) {
    return NodeAccess::getValuePtr<T>(N);
  }
  static const_pointer getValuePtr(const node_type *N) {
    return NodeAccess::getValuePtr<T>(N);
  }
};
} // end namespace ilist_detail

template <typename NodeTy> class ilist_sentinel : public ilist_node<NodeTy> {
public:
  ilist_sentinel() {
    ilist_node_base::initializeSentinel();
    reset();
  }

  void reset() {
    this->setPrev(this);
    this->setNext(this);
  }

  bool empty() const { return this == this->getPrev(); }
};

/// An ilist node that can access its parent list.
///
/// Requires \c NodeTy to have \a getParent() to find the parent node, and the
/// \c ParentTy to have \a getSublistAccess() to get a reference to the list.
template <typename NodeTy, typename ParentTy>
class ilist_node_with_parent : public ilist_node<NodeTy> {
protected:
  ilist_node_with_parent() = default;

private:
  /// Forward to NodeTy::getParent().
  ///
  /// Note: do not use the name "getParent()".  We want a compile error
  /// (instead of recursion) when the subclass fails to implement \a
  /// getParent().
  const ParentTy *getNodeParent() const {
    return static_cast<const NodeTy *>(this)->getParent();
  }

public:
  /// @name Adjacent Node Accessors
  /// @{
  /// \brief Get the previous node, or \c nullptr for the list head.
  NodeTy *getPrevNode() {
    // Should be separated to a reused function, but then we couldn't use auto
    // (and would need the type of the list).
    const auto &List =
        getNodeParent()->*(ParentTy::getSublistAccess((NodeTy *)nullptr));
    return List.getPrevNode(*static_cast<NodeTy *>(this));
  }
  /// \brief Get the previous node, or \c nullptr for the list head.
  const NodeTy *getPrevNode() const {
    return const_cast<ilist_node_with_parent *>(this)->getPrevNode();
  }

  /// \brief Get the next node, or \c nullptr for the list tail.
  NodeTy *getNextNode() {
    // Should be separated to a reused function, but then we couldn't use auto
    // (and would need the type of the list).
    const auto &List =
        getNodeParent()->*(ParentTy::getSublistAccess((NodeTy *)nullptr));
    return List.getNextNode(*static_cast<NodeTy *>(this));
  }
  /// \brief Get the next node, or \c nullptr for the list tail.
  const NodeTy *getNextNode() const {
    return const_cast<ilist_node_with_parent *>(this)->getNextNode();
  }
  /// @}
};

} // End llvm namespace

#endif
