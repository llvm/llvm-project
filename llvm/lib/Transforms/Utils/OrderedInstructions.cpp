//===-- OrderedInstructions.cpp - Instruction dominance function ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility to check dominance relation of 2 instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/OrderedInstructions.h"
using namespace llvm;

/// Given 2 instructions, use OrderedBasicBlock to check for dominance relation
/// if the instructions are in the same basic block, Otherwise, use dominator
/// tree.
bool OrderedInstructions::dominates(const Instruction *InstA,
                                    const Instruction *InstB) const {
  const BasicBlock *IBB = InstA->getParent();
  // Use ordered basic block to do dominance check in case the 2 instructions
  // are in the same basic block.
  if (IBB == InstB->getParent()) {
    auto OBB = OBBMap.find(IBB);
    if (OBB == OBBMap.end())
      OBB = OBBMap.insert({IBB, make_unique<OrderedBasicBlock>(IBB)}).first;
    return OBB->second->dominates(InstA, InstB);
  }
  return DT->dominates(InstA->getParent(), InstB->getParent());
}
