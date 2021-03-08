//===-- M68kTargetInfo.cpp - M68k Target Implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains M68k target initializer.
///
//===----------------------------------------------------------------------===//
#include "MCTargetDesc/M68kMCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

Target llvm::TheM68kTarget;

extern "C" void LLVMInitializeM68kTargetInfo() {
  RegisterTarget<Triple::m68k, /*HasJIT=*/true> X(
      TheM68kTarget, "m68k", "Motorola 68000 family", "M68k");
}
