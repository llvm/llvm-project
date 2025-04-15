//===- Use.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Use.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/User.h"

namespace llvm::sandboxir {

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

} // namespace llvm::sandboxir
