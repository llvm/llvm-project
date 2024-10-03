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
#include "llvm/SandboxIR/SandboxIRValues.def"
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
