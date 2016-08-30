//===- llvm/ADT/ilist_node_base.h - Intrusive List Node Base -----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_NODE_BASE_H
#define LLVM_ADT_ILIST_NODE_BASE_H

#include "llvm/ADT/PointerIntPair.h"

namespace llvm {

/// Base class for ilist nodes.
class ilist_node_base {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  PointerIntPair<ilist_node_base *, 1> PrevAndSentinel;
#else
  ilist_node_base *Prev = nullptr;
#endif
  ilist_node_base *Next = nullptr;

public:
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  void setPrev(ilist_node_base *Prev) { PrevAndSentinel.setPointer(Prev); }
  ilist_node_base *getPrev() const { return PrevAndSentinel.getPointer(); }

  bool isKnownSentinel() const { return PrevAndSentinel.getInt(); }
  void initializeSentinel() { PrevAndSentinel.setInt(true); }
#else
  void setPrev(ilist_node_base *Prev) { this->Prev = Prev; }
  ilist_node_base *getPrev() const { return Prev; }

  bool isKnownSentinel() const { return false; }
  void initializeSentinel() {}
#endif

  void setNext(ilist_node_base *Next) { this->Next = Next; }
  ilist_node_base *getNext() const { return Next; }
};

} // end namespace llvm

#endif // LLVM_ADT_ILIST_NODE_BASE_H
