//===-- XtensaTargetInfo.cpp - Xtensa Target Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetRegistry.h"

using namespace llvm;
namespace llvm {
Target TheXtensaTarget;
}
extern "C" void LLVMInitializeXtensaTargetInfo() {
  RegisterTarget<Triple::xtensa> X(TheXtensaTarget, "xtensa", "Xtensa 32",
                                   "XTENSA");
}
