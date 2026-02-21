//===-- ConnexTargetInfo.cpp - Connex Target Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Connex.h"
#include "TargetInfo/ConnexTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheConnexTarget() {
  static Target TheConnexTarget;
  return TheConnexTarget;
}

/* namespace llvm {
Target TheConnexTarget;
}
*/

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeConnexTargetInfo() {
  TargetRegistry::RegisterTarget(
      getTheConnexTarget(), "connex", "Connex", "Connex",
      [](Triple::ArchType) { return false; }, true);
}
