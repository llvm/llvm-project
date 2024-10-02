//===- User.cpp - The User class of Sandbox IR ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/User.h"
#include "llvm/SandboxIR/Context.h"

namespace llvm::sandboxir {

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

Use User::getOperandUseDefault(unsigned OpIdx, bool Verify) const {
  assert((!Verify || OpIdx < getNumOperands()) && "Out of bounds!");
  assert(isa<llvm::User>(Val) && "Non-users have no operands!");
  llvm::Use *LLVMUse;
  if (OpIdx != getNumOperands())
    LLVMUse = &cast<llvm::User>(Val)->getOperandUse(OpIdx);
  else
    LLVMUse = cast<llvm::User>(Val)->op_end();
  return Use(LLVMUse, const_cast<User *>(this), Ctx);
}

#ifndef NDEBUG
void User::verifyUserOfLLVMUse(const llvm::Use &Use) const {
  assert(Ctx.getValue(Use.getUser()) == this &&
         "Use not found in this SBUser's operands!");
}
#endif

bool User::classof(const Value *From) {
  switch (From->getSubclassID()) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return true;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/SandboxIR/Values.def"
  default:
    return false;
  }
}

void User::setOperand(unsigned OperandIdx, Value *Operand) {
  assert(isa<llvm::User>(Val) && "No operands!");
  Ctx.getTracker().emplaceIfTracking<UseSet>(getOperandUse(OperandIdx));
  // We are delegating to llvm::User::setOperand().
  cast<llvm::User>(Val)->setOperand(OperandIdx, Operand->Val);
}

bool User::replaceUsesOfWith(Value *FromV, Value *ToV) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto OpIdx : seq<unsigned>(0, getNumOperands())) {
      auto Use = getOperandUse(OpIdx);
      if (Use.get() == FromV)
        Tracker.emplaceIfTracking<UseSet>(Use);
    }
  }
  // We are delegating RUOW to LLVM IR's RUOW.
  return cast<llvm::User>(Val)->replaceUsesOfWith(FromV->Val, ToV->Val);
}

#ifndef NDEBUG
void User::dumpCommonHeader(raw_ostream &OS) const {
  Value::dumpCommonHeader(OS);
  // TODO: This is incomplete
}
#endif // NDEBUG

} // namespace llvm::sandboxir
