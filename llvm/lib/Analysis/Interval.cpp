//===- Interval.cpp - Interval class code ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the Interval class, which represents a
// partition of a control flow graph of some kind.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Interval.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

void Interval::print(raw_ostream &OS) const {
  OS << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";

  // Print out all of the basic blocks in the interval...
  for (const BasicBlock *Node : Nodes)
    OS << *Node << "\n";

  OS << "Interval Predecessors:\n";
  for (const BasicBlock *Predecessor : Predecessors)
    OS << *Predecessor << "\n";

  OS << "Interval Successors:\n";
  for (const BasicBlock *Successor : Successors)
    OS << *Successor << "\n";
}
