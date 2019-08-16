//===-- OrderedInstructions.cpp - Instruction dominance function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility to check dominance relation of 2 instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OrderedInstructions.h"
using namespace llvm;

bool OrderedInstructions::localDominates(const Instruction *InstA,
                                         const Instruction *InstB) const {
  assert(InstA->getParent() == InstB->getParent() &&
         "Instructions must be in the same basic block");

  const BasicBlock *IBB = InstA->getParent();
  auto OBB = OBBMap.find(IBB);
  if (OBB == OBBMap.end())
    OBB = OBBMap.insert({IBB, std::make_unique<OrderedBasicBlock>(IBB)}).first;
  return OBB->second->dominates(InstA, InstB);
}

/// Given 2 instructions, use OrderedBasicBlock to check for dominance relation
/// if the instructions are in the same basic block, Otherwise, use dominator
/// tree.
bool OrderedInstructions::dominates(const Instruction *InstA,
                                    const Instruction *InstB) const {
  // Use ordered basic block to do dominance check in case the 2 instructions
  // are in the same basic block.
  if (InstA->getParent() == InstB->getParent())
    return localDominates(InstA, InstB);
  return DT->dominates(InstA->getParent(), InstB->getParent());
}

bool OrderedInstructions::dfsBefore(const Instruction *InstA,
                                    const Instruction *InstB) const {
  // Use ordered basic block in case the 2 instructions are in the same basic
  // block.
  if (InstA->getParent() == InstB->getParent())
    return localDominates(InstA, InstB);

  DomTreeNode *DA = DT->getNode(InstA->getParent());
  DomTreeNode *DB = DT->getNode(InstB->getParent());
  return DA->getDFSNumIn() < DB->getDFSNumIn();
}
