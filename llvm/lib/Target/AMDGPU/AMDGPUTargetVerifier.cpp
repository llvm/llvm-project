//===-- AMDGPUTargetVerifier.cpp - AMDGPU -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target verifier interfaces that can be used for some
// validation of input to the system, and for checking that transformations
// haven't done something bad. In contrast to the Verifier or Lint, the
// TargetVerifier looks for constructions invalid to a particular target
// machine.
//
// To see what specifically is checked, look at an individual backend's
// TargetVerifier.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Check - We know that cond should be true, if not print an error message.
#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      TargetVerify::checkFailed(__VA_ARGS__);                                  \
    }                                                                          \
  } while (false)

namespace llvm {

class AMDGPUTargetVerify : public TargetVerify {
public:
  AMDGPUTargetVerify(Module *Mod) : TargetVerify(Mod) {}
  bool run(Function &F) override;
};

static bool IsValidInt(const Type *Ty) {
  return Ty->isIntegerTy(1) ||
         Ty->isIntegerTy(8) ||
         Ty->isIntegerTy(16) ||
         Ty->isIntegerTy(32) ||
         Ty->isIntegerTy(64) ||
         Ty->isIntegerTy(128);
}

bool AMDGPUTargetVerify::run(Function &F) {

  for (auto &BB : F) {

    for (auto &I : BB) {

      // Ensure integral types are valid: i8, i16, i32, i64, i128
      if (I.getType()->isIntegerTy())
        Check(IsValidInt(I.getType()), "Int type is invalid.", &I);
      for (unsigned i = 0; i < I.getNumOperands(); ++i)
        if (I.getOperand(i)->getType()->isIntegerTy())
          Check(IsValidInt(I.getOperand(i)->getType()),
                "Int type is invalid.", I.getOperand(i));

    }
  }

  dbgs() << MessagesStr.str();
  if (!MessagesStr.str().empty()) {
    IsValid = false;
    return false;
  }
  return true;
}

PreservedAnalyses AMDGPUTargetVerifierPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  auto *Mod = F.getParent();

  AMDGPUTargetVerify TV(Mod);
  TV.run(F);

  dbgs() << TV.MessagesStr.str();
  if (!TV.MessagesStr.str().empty()) {
    TV.IsValid = false;
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}

struct AMDGPUTargetVerifierLegacyPass : public FunctionPass {
  static char ID;

  std::unique_ptr<AMDGPUTargetVerify> TV;
  bool FatalErrors = false;

  AMDGPUTargetVerifierLegacyPass(bool FatalErrors)
      : FunctionPass(ID), FatalErrors(FatalErrors) {
    initializeAMDGPUTargetVerifierLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override {
    TV = std::make_unique<AMDGPUTargetVerify>(&M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (!TV->run(F)) {
      errs() << "in function " << F.getName() << '\n';
      if (FatalErrors)
        report_fatal_error("broken function found, compilation aborted!");
      else
        errs() << "broken function found, compilation aborted!\n";
    }
    return false;
  }

  bool doFinalization(Module &M) override {
    bool IsValid = true;
    for (Function &F : M)
      if (F.isDeclaration())
        IsValid &= TV->run(F);

    if (!IsValid) {
      if (FatalErrors)
        report_fatal_error("broken module found, compilation aborted!");
      else
        errs() << "broken module found, compilation aborted!\n";
    }
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
char AMDGPUTargetVerifierLegacyPass::ID = 0;
FunctionPass *createAMDGPUTargetVerifierLegacyPass(bool FatalErrors) {
  return new AMDGPUTargetVerifierLegacyPass(FatalErrors);
}
} // namespace llvm
INITIALIZE_PASS(AMDGPUTargetVerifierLegacyPass, "amdgpu-tgtverifier",
                "AMDGPU Target Verifier", false, false)
