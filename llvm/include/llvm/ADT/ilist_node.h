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

namespace llvm {

template<typename NodeTy>
struct ilist_traits;
template <typename NodeTy> struct ilist_embedded_sentinel_traits;
template <typename NodeTy> struct ilist_half_embedded_sentinel_traits;

/// ilist_half_node - Base class that provides prev services for sentinels.
///
template<typename NodeTy>
class ilist_half_node {
  friend struct ilist_traits<NodeTy>;
  friend struct ilist_half_embedded_sentinel_traits<NodeTy>;
  NodeTy *Prev;
protected:
  NodeTy *getPrev() { return Prev; }
  const NodeTy *getPrev() const { return Prev; }
  void setPrev(NodeTy *P) { Prev = P; }
  ilist_half_node() : Prev(nullptr) {}
};

template<typename NodeTy>
struct ilist_nextprev_traits;

template <typename NodeTy> class ilist_iterator;

/// ilist_node - Base class that provides next/prev services for nodes
/// that use ilist_nextprev_traits or ilist_default_traits.
///
template<typename NodeTy>
class ilist_node : private ilist_half_node<NodeTy> {
  friend struct ilist_nextprev_traits<NodeTy>;
  friend struct ilist_traits<NodeTy>;
  friend struct ilist_half_embedded_sentinel_traits<NodeTy>;
  friend struct ilist_embedded_sentinel_traits<NodeTy>;
  NodeTy *Next;
  NodeTy *getNext() { return Next; }
  const NodeTy *getNext() const { return Next; }
  void setNext(NodeTy *N) { Next = N; }
protected:
  ilist_node() : Next(nullptr) {}

public:
  ilist_iterator<NodeTy> getIterator() {
    // FIXME: Stop downcasting to create the iterator (potential UB).
    return ilist_iterator<NodeTy>(static_cast<NodeTy *>(this));
  }
  ilist_iterator<const NodeTy> getIterator() const {
    // FIXME: Stop downcasting to create the iterator (potential UB).
    return ilist_iterator<const NodeTy>(static_cast<const NodeTy *>(this));
  }
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
