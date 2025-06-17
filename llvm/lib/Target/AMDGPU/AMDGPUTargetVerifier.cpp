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

static bool isShader(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS_Chain:
  case CallingConv::AMDGPU_CS_ChainPreserve:
  case CallingConv::AMDGPU_CS:
    return true;
  default:
    return false;
  }
}

bool AMDGPUTargetVerify::run(Function &F) {
  // Ensure shader calling convention returns void
  if (isShader(F.getCallingConv()))
    Check(F.getReturnType() == Type::getVoidTy(F.getContext()),
          "Shaders must return void");

  for (auto &BB : F) {

    for (auto &I : BB) {

      if (auto *CI = dyn_cast<CallInst>(&I)) {
        // Ensure no kernel to kernel calls.
        CallingConv::ID CalleeCC = CI->getCallingConv();
        if (CalleeCC == CallingConv::AMDGPU_KERNEL) {
          CallingConv::ID CallerCC =
              CI->getParent()->getParent()->getCallingConv();
          Check(CallerCC != CallingConv::AMDGPU_KERNEL,
                "A kernel may not call a kernel", CI->getParent()->getParent());
        }

        // Ensure chain intrinsics are followed by unreachables.
        if (CI->getIntrinsicID() == Intrinsic::amdgcn_cs_chain)
          Check(isa_and_present<UnreachableInst>(CI->getNextNode()),
                "llvm.amdgcn.cs.chain must be followed by unreachable", CI);
      }
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
