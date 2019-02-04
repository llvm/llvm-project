//===-- PowerPCTargetInfo.cpp - PowerPC Target Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target &llvm::getThePPC32Target() {
  static Target ThePPC32Target;
  return ThePPC32Target;
}
Target &llvm::getThePPC64Target() {
  static Target ThePPC64Target;
  return ThePPC64Target;
}
Target &llvm::getThePPC64LETarget() {
  static Target ThePPC64LETarget;
  return ThePPC64LETarget;
}

extern "C" void LLVMInitializePowerPCTargetInfo() {
  RegisterTarget<Triple::ppc, /*HasJIT=*/true> X(getThePPC32Target(), "ppc32",
                                                 "PowerPC 32", "PPC");

  RegisterTarget<Triple::ppc64, /*HasJIT=*/true> Y(getThePPC64Target(), "ppc64",
                                                   "PowerPC 64", "PPC");

  RegisterTarget<Triple::ppc64le, /*HasJIT=*/true> Z(
      getThePPC64LETarget(), "ppc64le", "PowerPC 64 LE", "PPC");
}
