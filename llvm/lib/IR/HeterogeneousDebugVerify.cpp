//===--- HeterogeneousDebugVerify.h - Strip above -O0 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Strip heterogeneous debug info at higher optimization levels for both
/// the new and legacy pass managers
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/HeterogeneousDebugVerify.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

using namespace llvm;

namespace {

cl::opt<bool> DisableHeterogeneousDebugVerify(
    "disable-heterogeneous-debug-verify",
    cl::desc("Do not strip heterogeneous debug metadata at optimization levels "
             "other than -O0"),
    cl::init(false));

static int DK_IncompatibleHeterogeneousDebug =
    getNextAvailablePluginDiagnosticKind();

struct DiagnosticInfoRemovingIncompatibleHeterogeneousDebug
    : public DiagnosticInfo {
  DiagnosticInfoRemovingIncompatibleHeterogeneousDebug(const Module &M)
      : DiagnosticInfo(DiagnosticKind(DK_IncompatibleHeterogeneousDebug),
                       DS_Warning),
        M(M) {}
  void print(DiagnosticPrinter &DP) const override {
    DP << M.getName()
       << ": removing heterogeneous debug metadata while compiling above -O0";
  }
  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_IncompatibleHeterogeneousDebug;
  }
  const Module &M;
};

constexpr unsigned RetainedNodesOpIdx = 7;

static bool maybeStrip(Module &M, CodeGenOpt::Level OptLevel,
                       bool IsNPM = false) {
  if (DisableHeterogeneousDebugVerify || !isHeterogeneousDebug(M))
    return false;
  if (OptLevel == CodeGenOpt::None && !IsNPM)
    return false;
  M.getContext().diagnose(
      DiagnosticInfoRemovingIncompatibleHeterogeneousDebug(M));
  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.retainedNodes"))
    M.eraseNamedMetadata(NMD);
  for (Function &F : M) {
    if (DISubprogram *S = F.getSubprogram())
      S->replaceOperandWith(RetainedNodesOpIdx,
                            MDTuple::get(F.getContext(), {}));
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB))
        if (isa<DbgDefKillIntrinsic>(&I))
          I.eraseFromParent();
  }
  for (auto &GV : M.globals())
    GV.eraseMetadata(M.getContext().getMDKindID("dbg.def"));
  M.setModuleFlag(
      llvm::Module::Warning, "Debug Info Version",
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                                               DEBUG_METADATA_VERSION)));
  return true;
}

class HeterogeneousDebugVerifyLegacy : public ModulePass {
  const CodeGenOpt::Level OptLevel;

public:
  static char ID;
  HeterogeneousDebugVerifyLegacy(CodeGenOpt::Level OptLevel)
      : ModulePass(ID), OptLevel(OptLevel) {}

  bool doInitialization(Module &M) override { return maybeStrip(M, OptLevel); }
  bool runOnModule(Module &M) override { return false; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  StringRef getPassName() const override {
    return "Verify Heterogeneous Debug Preconditions";
  }
};

} // namespace

char HeterogeneousDebugVerifyLegacy::ID = 0;
INITIALIZE_PASS(HeterogeneousDebugVerifyLegacy,
                "heterogeneous-debug-verify-legacy-pass",
                "Verify heterogeneous debug preconditions", false, true)

ModulePass *
llvm::createHeterogeneousDebugVerifyLegacyPass(CodeGenOpt::Level OptLevel) {
  return new HeterogeneousDebugVerifyLegacy(OptLevel);
}

namespace llvm {

HeterogeneousDebugVerify::HeterogeneousDebugVerify(CodeGenOpt::Level OptLevel)
    : OptLevel(OptLevel) {}
PreservedAnalyses HeterogeneousDebugVerify::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  (void)maybeStrip(M, OptLevel, /*IsNPM=*/true);
  return PreservedAnalyses::all();
}

} // namespace llvm
