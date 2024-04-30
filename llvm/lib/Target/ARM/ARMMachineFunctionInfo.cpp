//===-- ARMMachineFunctionInfo.cpp - ARM machine function info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"

using namespace llvm;

void ARMFunctionInfo::anchor() {}

yaml::ARMFunctionInfo::ARMFunctionInfo(const llvm::ARMFunctionInfo &MFI)
    : LRSpilled(MFI.isLRSpilled()) {}

void yaml::ARMFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<ARMFunctionInfo>::mapping(YamlIO, *this);
}

void ARMFunctionInfo::initializeBaseYamlFields(
    const yaml::ARMFunctionInfo &YamlMFI) {
  LRSpilled = YamlMFI.LRSpilled;
}

static bool GetBranchTargetEnforcement(const Function &F,
                                       const ARMSubtarget *Subtarget) {
  if (!Subtarget->isMClass() || !Subtarget->hasV7Ops())
    return false;

  if (!F.hasFnAttribute("branch-target-enforcement")) {
    if (const auto *BTE = mdconst::extract_or_null<ConstantInt>(
            F.getParent()->getModuleFlag("branch-target-enforcement")))
      return BTE->getZExtValue();
    return false;
  }

  const StringRef BTIEnable =
      F.getFnAttribute("branch-target-enforcement").getValueAsString();
  assert(BTIEnable == "true" || BTIEnable == "false");
  return BTIEnable == "true";
}

// The pair returns values for the ARMFunctionInfo members
// SignReturnAddress and SignReturnAddressAll respectively.
static std::pair<bool, bool> GetSignReturnAddress(const Function &F) {
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

ARMFunctionInfo::ARMFunctionInfo(const Function &F,
                                 const ARMSubtarget *Subtarget)
    : isThumb(Subtarget->isThumb()), hasThumb2(Subtarget->hasThumb2()),
      IsCmseNSEntry(F.hasFnAttribute("cmse_nonsecure_entry")),
      IsCmseNSCall(F.hasFnAttribute("cmse_nonsecure_call")),
      BranchTargetEnforcement(GetBranchTargetEnforcement(F, Subtarget)) {
  if (Subtarget->isMClass() && Subtarget->hasV7Ops())
    std::tie(SignReturnAddress, SignReturnAddressAll) = GetSignReturnAddress(F);
}

MachineFunctionInfo *
ARMFunctionInfo::clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
                       const DenseMap<MachineBasicBlock *, MachineBasicBlock *>
                           &Src2DstMBB) const {
  return DestMF.cloneInfo<ARMFunctionInfo>(*this);
}
