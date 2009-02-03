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
struct ilist_nextprev_traits;

/// ilist_node - Base class that provides next/prev services for nodes
/// that use ilist_nextprev_traits or ilist_default_traits.
///
template<typename NodeTy>
class ilist_node {
private:
  friend struct ilist_nextprev_traits<NodeTy>;
  NodeTy *Prev, *Next;
  NodeTy *getPrev() { return Prev; }
  NodeTy *getNext() { return Next; }
  const NodeTy *getPrev() const { return Prev; }
  const NodeTy *getNext() const { return Next; }
  void setPrev(NodeTy *N) { Prev = N; }
  void setNext(NodeTy *N) { Next = N; }
protected:
  ilist_node() : Prev(0), Next(0) {}
};

} // End llvm namespace

#endif
