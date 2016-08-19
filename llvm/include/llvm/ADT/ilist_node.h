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

#include <llvm/ADT/PointerIntPair.h>

namespace llvm {

template<typename NodeTy>
struct ilist_traits;
template <typename NodeTy> struct ilist_embedded_sentinel_traits;
template <typename NodeTy> struct ilist_half_embedded_sentinel_traits;

/// Base class for ilist nodes.
struct ilist_node_base {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  PointerIntPair<ilist_node_base *, 1> PrevAndSentinel;

  void setPrev(ilist_node_base *Prev) { PrevAndSentinel.setPointer(Prev); }
  ilist_node_base *getPrev() const { return PrevAndSentinel.getPointer(); }

  bool isKnownSentinel() const { return PrevAndSentinel.getInt(); }
  void initializeSentinel() { PrevAndSentinel.setInt(true); }
#else
  ilist_node_base *Prev = nullptr;

  void setPrev(ilist_node_base *Prev) { this->Prev = Prev; }
  ilist_node_base *getPrev() const { return Prev; }

  bool isKnownSentinel() const { return false; }
  void initializeSentinel() {}
#endif

  ilist_node_base *Next = nullptr;
};

struct ilist_node_access;
template <typename NodeTy> class ilist_iterator;
template <typename NodeTy> class ilist_sentinel;

/// Templated wrapper class.
template <typename NodeTy> class ilist_node : ilist_node_base {
  friend struct ilist_node_access;
  friend struct ilist_traits<NodeTy>;
  friend struct ilist_half_embedded_sentinel_traits<NodeTy>;
  friend struct ilist_embedded_sentinel_traits<NodeTy>;
  friend class ilist_iterator<NodeTy>;
  friend class ilist_sentinel<NodeTy>;

protected:
  ilist_node() = default;

private:
  ilist_node *getPrev() {
    return static_cast<ilist_node *>(ilist_node_base::getPrev());
  }
  ilist_node *getNext() { return static_cast<ilist_node *>(Next); }

  const ilist_node *getPrev() const {
    return static_cast<ilist_node *>(ilist_node_base::getPrev());
  }
  const ilist_node *getNext() const { return static_cast<ilist_node *>(Next); }

  void setPrev(ilist_node *N) { ilist_node_base::setPrev(N); }
  void setNext(ilist_node *N) { Next = N; }

public:
  ilist_iterator<NodeTy> getIterator() { return ilist_iterator<NodeTy>(*this); }
  ilist_iterator<const NodeTy> getIterator() const {
    return ilist_iterator<const NodeTy>(*this);
  }

  using ilist_node_base::isKnownSentinel;
};

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
