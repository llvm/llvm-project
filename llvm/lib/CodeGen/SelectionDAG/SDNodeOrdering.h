//===-- llvm/CodeGen/SDNodeOrdering.h - SDNode Ordering ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDNodeOrdering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDNODEORDERING_H
#define LLVM_CODEGEN_SDNODEORDERING_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class SDNode;

/// SDNodeOrdering - Maps a unique (monotonically increasing) value to each
/// SDNode that roughly corresponds to the ordering of the original LLVM
/// instruction. This is used for turning off scheduling, because we'll forgo
/// the normal scheduling algorithms and output the instructions according to
/// this ordering.
class SDNodeOrdering {
  DenseMap<const SDNode*, unsigned> OrderMap;

  void operator=(const SDNodeOrdering&) LLVM_DELETED_FUNCTION;
  SDNodeOrdering(const SDNodeOrdering&) LLVM_DELETED_FUNCTION;
public:
  SDNodeOrdering() {}

  void add(const SDNode *Node, unsigned O) {
    OrderMap[Node] = O;
  }
  void remove(const SDNode *Node) {
    DenseMap<const SDNode*, unsigned>::iterator Itr = OrderMap.find(Node);
    if (Itr != OrderMap.end())
      OrderMap.erase(Itr);
  }
  void clear() {
    OrderMap.clear();
  }
  unsigned getOrder(const SDNode *Node) {
    return OrderMap[Node];
  }
};

} // end llvm namespace

#endif
