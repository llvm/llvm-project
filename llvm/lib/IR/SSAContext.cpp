//===- SSAContext.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a specialization of the GenericSSAContext<X>
/// template class for LLVM IR.
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/SSAContext.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

BasicBlock *SSAContext::getEntryBlock(Function &F) {
  return &F.getEntryBlock();
}

void SSAContext::setFunction(Function &Fn) { F = &Fn; }

Printable SSAContext::print(Value *V) const {
  return Printable([V](raw_ostream &Out) { V->print(Out); });
}

Printable SSAContext::print(Instruction *Inst) const {
  return print(cast<Value>(Inst));
}

Printable SSAContext::print(BasicBlock *BB) const {
  if (BB->hasName())
    return Printable([BB](raw_ostream &Out) { Out << BB->getName(); });

  return Printable([BB](raw_ostream &Out) {
    ModuleSlotTracker MST{BB->getParent()->getParent(), false};
    MST.incorporateFunction(*BB->getParent());
    Out << MST.getLocalSlot(BB);
  });
}
