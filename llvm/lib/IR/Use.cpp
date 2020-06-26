//===-- Use.cpp - Implement the Use class ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include <new>

namespace llvm {

void Use::swap(Use &RHS) {
  if (Val == RHS.Val)
    return;

  if (Val)
    removeFromList();

  Value *OldVal = Val;
  if (RHS.Val) {
    RHS.removeFromList();
    Val = RHS.Val;
    Val->addUse(*this);
  } else {
    Val = nullptr;
  }

  if (OldVal) {
    RHS.Val = OldVal;
    RHS.Val->addUse(RHS);
  } else {
    RHS.Val = nullptr;
  }
}

unsigned Use::getOperandNo() const {
  return this - getUser()->op_begin();
}

void Use::zap(Use *Start, const Use *Stop, bool del) {
  while (Start != Stop)
    (--Stop)->~Use();
  if (del)
    ::operator delete(Start);
}

} // namespace llvm
