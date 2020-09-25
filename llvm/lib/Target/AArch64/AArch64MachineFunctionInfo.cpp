//=- AArch64MachineFunctionInfo.cpp - AArch64 Machine Function Info ---------=//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements AArch64-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#include "AArch64MachineFunctionInfo.h"
#include "AArch64InstrInfo.h"
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

using namespace llvm;

yaml::AArch64FunctionInfo::AArch64FunctionInfo(
    const llvm::AArch64FunctionInfo &MFI)
    : HasRedZone(MFI.hasRedZone()) {}

void yaml::AArch64FunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<AArch64FunctionInfo>::mapping(YamlIO, *this);
}

void AArch64FunctionInfo::initializeBaseYamlFields(
    const yaml::AArch64FunctionInfo &YamlMFI) {
  if (YamlMFI.HasRedZone.hasValue())
    HasRedZone = YamlMFI.HasRedZone;
}

static std::pair<bool, bool> GetSignReturnAddress(const Function &F) {
  // The function should be signed in the following situations:
  // - sign-return-address=all
  // - sign-return-address=non-leaf and the functions spills the LR
  if (!F.hasFnAttribute("sign-return-address")) {
    const Module &M = *F.getParent();
    if (const auto *Sign = mdconst::extract_or_null<ConstantInt>(
            M.getModuleFlag("sign-return-address"))) {
      if (Sign->getZExtValue()) {
        if (const auto *All = mdconst::extract_or_null<ConstantInt>(
                M.getModuleFlag("sign-return-address-all")))
          return {true, All->getZExtValue()};
        return {true, false};
      }
    }
    return {false, false};
  }

  StringRef Scope = F.getFnAttribute("sign-return-address").getValueAsString();
  if (Scope.equals("none"))
    return {false, false};

  if (Scope.equals("all"))
    return {true, true};

  assert(Scope.equals("non-leaf"));
  return {true, false};
}

static bool ShouldSignWithBKey(const Function &F) {
  if (!F.hasFnAttribute("sign-return-address-key")) {
    if (const auto *BKey = mdconst::extract_or_null<ConstantInt>(
            F.getParent()->getModuleFlag("sign-return-address-with-bkey")))
      return BKey->getZExtValue();
    return false;
  }

  const StringRef Key =
      F.getFnAttribute("sign-return-address-key").getValueAsString();
  assert(Key.equals_lower("a_key") || Key.equals_lower("b_key"));
  return Key.equals_lower("b_key");
}

AArch64FunctionInfo::AArch64FunctionInfo(MachineFunction &MF) : MF(MF) {
  // If we already know that the function doesn't have a redzone, set
  // HasRedZone here.
  if (MF.getFunction().hasFnAttribute(Attribute::NoRedZone))
    HasRedZone = false;

  const Function &F = MF.getFunction();
  std::tie(SignReturnAddress, SignReturnAddressAll) = GetSignReturnAddress(F);
  SignWithBKey = ShouldSignWithBKey(F);

  if (!F.hasFnAttribute("branch-target-enforcement")) {
    if (const auto *BTE = mdconst::extract_or_null<ConstantInt>(
            F.getParent()->getModuleFlag("branch-target-enforcement")))
      BranchTargetEnforcement = BTE->getZExtValue();
    return;
  }

  const StringRef BTIEnable = F.getFnAttribute("branch-target-enforcement").getValueAsString();
  assert(BTIEnable.equals_lower("true") || BTIEnable.equals_lower("false"));
  BranchTargetEnforcement = BTIEnable.equals_lower("true");
}

bool AArch64FunctionInfo::shouldSignReturnAddress(bool SpillsLR) const {
  if (!SignReturnAddress)
    return false;
  if (SignReturnAddressAll)
    return true;
  return SpillsLR;
}

bool AArch64FunctionInfo::shouldSignReturnAddress() const {
  return shouldSignReturnAddress(llvm::any_of(
      MF.getFrameInfo().getCalleeSavedInfo(),
      [](const auto &Info) { return Info.getReg() == AArch64::LR; }));
}
