//===- llvm/ADT/ilist_base.h - Intrusive List Base ---------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_BASE_H
#define LLVM_ADT_ILIST_BASE_H

#include "llvm/ADT/ilist_node_base.h"
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace llvm {

/// Implementations of list algorithms using ilist_node_base.
class ilist_base {
public:
  static void insertBeforeImpl(ilist_node_base &Next, ilist_node_base &N) {
    ilist_node_base &Prev = *Next.getPrev();
    N.setNext(&Next);
    N.setPrev(&Prev);
    Prev.setNext(&N);
    Next.setPrev(&N);
  }

  static void removeImpl(ilist_node_base &N) {
    ilist_node_base *Prev = N.getPrev();
    ilist_node_base *Next = N.getNext();
    Next->setPrev(Prev);
    Prev->setNext(Next);

    // Not strictly necessary, but helps catch a class of bugs.
    N.setPrev(nullptr);
    N.setNext(nullptr);
  }

  static void transferBeforeImpl(ilist_node_base &Next, ilist_node_base &First,
                                 ilist_node_base &Last) {
    assert(&Next != &Last && "Should be checked by callers");
    assert(&First != &Last && "Should be checked by callers");
    // Position cannot be contained in the range to be transferred.
    assert(&Next != &First &&
           // Check for the most common mistake.
           "Insertion point can't be one of the transferred nodes");

    ilist_node_base &Final = *Last.getPrev();

    // Detach from old list/position.
    First.getPrev()->setNext(&Last);
    Last.setPrev(First.getPrev());

    // Splice [First, Final] into its new list/position.
    ilist_node_base &Prev = *Next.getPrev();
    Final.setNext(&Next);
    First.setPrev(&Prev);
    Prev.setNext(&First);
    Next.setPrev(&Final);
  }

  template <class T> static void insertBefore(T &Next, T &N) {
    insertBeforeImpl(Next, N);
  }

  template <class T> static void remove(T &N) { removeImpl(N); }

  template <class T> static void transferBefore(T &Next, T &First, T &Last) {
    transferBeforeImpl(Next, First, Last);
  }
};

} // end namespace llvm

#endif // LLVM_ADT_ILIST_BASE_H
