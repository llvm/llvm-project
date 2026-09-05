//===-- PISATargetInfo.cpp - PISA Target Implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/PISATargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getThePISATarget() {
  static Target ThePISATarget;
  return ThePISATarget;
}

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePISATargetInfo() {
  RegisterTarget<Triple::pisa> Y(getThePISATarget(), "pisa", "PISA 64-bit",
                                 "PISA");
}
