//===--- AArch64MachineModuleInfo.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AArch64 Machine Module Info.
///
//
//===----------------------------------------------------------------------===//

#include "AArch64MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

namespace llvm {

AArch64MachineModuleInfo::AArch64MachineModuleInfo(const MachineModuleInfo &MMI)
    : MachineModuleInfoELF(MMI) {
  const Module *M = MMI.getModule();
  const auto *Flag = mdconst::extract_or_null<ConstantInt>(
      M->getModuleFlag("ptrauth-sign-personality"));
  if (Flag && Flag->getZExtValue() == 1)
    HasSignedPersonality = true;
  else
    HasSignedPersonality = false;
}

} // end namespace llvm
