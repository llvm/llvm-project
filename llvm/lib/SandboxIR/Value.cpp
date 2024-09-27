//===- Value.cpp - The Value class of Sandbox IR --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Value.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <sstream>

namespace llvm::sandboxir {

Value::Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx)
    : SubclassID(SubclassID), Val(Val), Ctx(Ctx) {
#ifndef NDEBUG
  UID = Ctx.getNumValues();
#endif
}

Value::use_iterator Value::use_begin() {
  llvm::Use *LLVMUse = nullptr;
  if (Val->use_begin() != Val->use_end())
    LLVMUse = &*Val->use_begin();
  User *User = LLVMUse != nullptr ? cast_or_null<sandboxir::User>(Ctx.getValue(
                                        Val->use_begin()->getUser()))
                                  : nullptr;
  return use_iterator(Use(LLVMUse, User, Ctx));
}

Value::user_iterator Value::user_begin() {
  auto UseBegin = Val->use_begin();
  auto UseEnd = Val->use_end();
  bool AtEnd = UseBegin == UseEnd;
  llvm::Use *LLVMUse = AtEnd ? nullptr : &*UseBegin;
  User *User =
      AtEnd ? nullptr
            : cast_or_null<sandboxir::User>(Ctx.getValue(&*LLVMUse->getUser()));
  return user_iterator(Use(LLVMUse, User, Ctx), UseToUser());
}

unsigned Value::getNumUses() const { return range_size(Val->users()); }

Type *Value::getType() const { return Ctx.getType(Val->getType()); }

void Value::replaceUsesWithIf(
    Value *OtherV, llvm::function_ref<bool(const Use &)> ShouldReplace) {
  assert(getType() == OtherV->getType() && "Can't replace with different type");
  llvm::Value *OtherVal = OtherV->Val;
  // We are delegating RUWIf to LLVM IR's RUWIf.
  Val->replaceUsesWithIf(
      OtherVal, [&ShouldReplace, this](llvm::Use &LLVMUse) -> bool {
        User *DstU = cast_or_null<User>(Ctx.getValue(LLVMUse.getUser()));
        if (DstU == nullptr)
          return false;
        Use UseToReplace(&LLVMUse, DstU, Ctx);
        if (!ShouldReplace(UseToReplace))
          return false;
        Ctx.getTracker().emplaceIfTracking<UseSet>(UseToReplace);
        return true;
      });
}

void Value::replaceAllUsesWith(Value *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with Value of different type!");
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto Use : uses())
      Tracker.track(std::make_unique<UseSet>(Use));
  }
  // We are delegating RAUW to LLVM IR's RAUW.
  Val->replaceAllUsesWith(Other->Val);
}

#ifndef NDEBUG
std::string Value::getUid() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void Value::dumpCommonHeader(raw_ostream &OS) const {
  OS << getUid() << " " << getSubclassIDStr(SubclassID) << " ";
}

void Value::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";
}

void Value::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void Value::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getUid() << " (" << getSubclassIDStr(SubclassID) << ")";
}

void Value::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void Value::dump() const {
  dumpOS(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
