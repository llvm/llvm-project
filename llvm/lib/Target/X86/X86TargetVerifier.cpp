//===-- X86TargetVerifier.cpp - X86 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the X86 implementation of the target-dependent verifier. It is the
// "TargetVerify" half of the verification the (target-independent)
// TargetVerifierPass runs for X86 modules; the other half is the generic IR
// verifier.
//
//===----------------------------------------------------------------------===//

#include "X86.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetVerifier.h"

using namespace llvm;

namespace {

class X86TargetVerify : public TargetVerify {
public:
  X86TargetVerify(Module *Mod) : TargetVerify(Mod) {}
  bool run(Function &F) override;
};

bool X86TargetVerify::run(Function &F) {
  IsValid = true;
  return IsValid;
}

} // anonymous namespace

namespace llvm {
TargetVerify *createX86TargetVerify(Module &M) {
  return new X86TargetVerify(&M);
}
} // namespace llvm
