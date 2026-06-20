//===-- TargetVerifier.cpp - Target-dependent IR Verifier ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent dispatcher for target-dependent
// IR verification. See llvm/Target/TargetVerifier.h for the design.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetVerifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// The registry maps Triple::ArchType (stored as unsigned, since there is no
// DenseMapInfo for the enum) to the factory that builds that target's
// TargetVerify. It is populated by backends from LLVMInitialize<T>Target().
static DenseMap<unsigned, TargetVerifyFactory> &getTargetVerifyRegistry() {
  static DenseMap<unsigned, TargetVerifyFactory> Registry;
  return Registry;
}

void llvm::registerTargetVerify(Triple::ArchType Arch,
                                TargetVerifyFactory Factory) {
  getTargetVerifyRegistry()[static_cast<unsigned>(Arch)] = Factory;
}

const TargetVerifyFactory *llvm::getTargetVerify(Triple::ArchType Arch) {
  DenseMap<unsigned, TargetVerifyFactory> &Registry = getTargetVerifyRegistry();
  auto It = Registry.find(static_cast<unsigned>(Arch));
  if (It == Registry.end())
    return nullptr;
  return &It->second;
}

// Run \p M's registered TargetVerify (if any) over every defined function,
// emitting any diagnostics to errs(). Returns true if a function is invalid for
// the target. This is only the target-specific layer; it does not run the
// generic IR verifier.
static bool runTargetVerify(Module &M) {
  const TargetVerifyFactory *Factory =
      getTargetVerify(M.getTargetTriple().getArch());
  if (!Factory)
    return false;

  std::unique_ptr<TargetVerify> TV{(*Factory)(M)};
  if (!TV)
    return false;

  bool Broken = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (!TV->run(F))
      Broken = true;
  }
  TV->MessagesStr.flush();
  if (!TV->Messages.empty())
    errs() << TV->Messages;
  return Broken;
}

PreservedAnalyses TargetVerifierPass::run(Module &M, ModuleAnalysisManager &AM) {
  // Use the module triple to decide which target-dependent verification to run.
  // If the target did not register a verifier, there is nothing target-specific
  // to do, so skip everything (including the generic verifier) and stay a no-op.
  if (!getTargetVerify(M.getTargetTriple().getArch()))
    return PreservedAnalyses::all();

  bool Broken = false;

  // (1) Run the generic IR verifier. After the VerifierAMDGPU split, the
  // generic verifier dispatches to each target's IR-level checks (e.g. the
  // verifyAMDGPU* routines in VerifierAMDGPU.cpp for AMDGPU modules), so this
  // is how the target's "<Target>Verifier" runs.
  Broken |= verifyModule(M, &errs());

  // (2) Run the target's subtarget/feature-dependent verifier (TargetVerify).
  Broken |= runTargetVerify(M);

  if (Broken) {
    if (FatalErrors)
      report_fatal_error("broken module found, compilation aborted!");
    errs() << "broken module found\n";
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}

namespace {
/// Legacy-PM wrapper that runs a module's registered TargetVerify inside the
/// codegen pipeline (e.g. `llc -verify-target`). The generic IR verifier runs
/// as a separate pass there, so this performs only the target-specific checks.
class TargetVerifierLegacyPass : public ModulePass {
  bool FatalErrors;

public:
  static char ID;

  explicit TargetVerifierLegacyPass(bool FatalErrors = true)
      : ModulePass(ID), FatalErrors(FatalErrors) {
    initializeTargetVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (runTargetVerify(M) && FatalErrors)
      report_fatal_error("broken module found, compilation aborted!");
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char TargetVerifierLegacyPass::ID = 0;

INITIALIZE_PASS(TargetVerifierLegacyPass, "target-verifier", "Target Verifier",
                false, false)

ModulePass *llvm::createTargetVerifierPass(bool FatalErrors) {
  return new TargetVerifierLegacyPass(FatalErrors);
}
