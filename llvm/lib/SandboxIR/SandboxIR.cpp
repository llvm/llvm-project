//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/SandboxIR/Argument.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/User.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm::sandboxir;

Value *Use::get() const { return Ctx->getValue(LLVMUse->get()); }

void Use::set(Value *V) {
  Ctx->getTracker().emplaceIfTracking<UseSet>(*this);
  LLVMUse->set(V->Val);
}

unsigned Use::getOperandNo() const { return Usr->getUseOperandNo(*this); }

void Use::swap(Use &OtherUse) {
  Ctx->getTracker().emplaceIfTracking<UseSwap>(*this, OtherUse);
  LLVMUse->swap(*OtherUse.LLVMUse);
}

#ifndef NDEBUG
void Use::dumpOS(raw_ostream &OS) const {
  Value *Def = nullptr;
  if (LLVMUse == nullptr)
    OS << "<null> LLVM Use! ";
  else
    Def = Ctx->getValue(LLVMUse->get());
  OS << "Def:  ";
  if (Def == nullptr)
    OS << "NULL";
  else
    OS << *Def;
  OS << "\n";

  OS << "User: ";
  if (Usr == nullptr)
    OS << "NULL";
  else
    OS << *Usr;
  OS << "\n";

  OS << "OperandNo: ";
  if (Usr == nullptr)
    OS << "N/A";
  else
    OS << getOperandNo();
  OS << "\n";
}

void Use::dump() const { dumpOS(dbgs()); }
#endif // NDEBUG

Use OperandUseIterator::operator*() const { return Use; }

OperandUseIterator &OperandUseIterator::operator++() {
  assert(Use.LLVMUse != nullptr && "Already at end!");
  User *User = Use.getUser();
  Use = User->getOperandUseInternal(Use.getOperandNo() + 1, /*Verify=*/false);
  return *this;
}

UserUseIterator &UserUseIterator::operator++() {
  // Get the corresponding llvm::Use, get the next in the list, and update the
  // sandboxir::Use.
  llvm::Use *&LLVMUse = Use.LLVMUse;
  assert(LLVMUse != nullptr && "Already at end!");
  LLVMUse = LLVMUse->getNext();
  if (LLVMUse == nullptr) {
    Use.Usr = nullptr;
    return *this;
  }
  auto *Ctx = Use.Ctx;
  auto *LLVMUser = LLVMUse->getUser();
  Use.Usr = cast_or_null<sandboxir::User>(Ctx->getValue(LLVMUser));
  return *this;
}

OperandUseIterator OperandUseIterator::operator+(unsigned Num) const {
  sandboxir::Use U = Use.getUser()->getOperandUseInternal(
      Use.getOperandNo() + Num, /*Verify=*/true);
  return OperandUseIterator(U);
}

OperandUseIterator OperandUseIterator::operator-(unsigned Num) const {
  assert(Use.getOperandNo() >= Num && "Out of bounds!");
  sandboxir::Use U = Use.getUser()->getOperandUseInternal(
      Use.getOperandNo() - Num, /*Verify=*/true);
  return OperandUseIterator(U);
}

int OperandUseIterator::operator-(const OperandUseIterator &Other) const {
  int ThisOpNo = Use.getOperandNo();
  int OtherOpNo = Other.Use.getOperandNo();
  return ThisOpNo - OtherOpNo;
}
