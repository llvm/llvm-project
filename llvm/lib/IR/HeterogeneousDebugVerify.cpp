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

static DICompileUnit *getCUForScope(DIScope *S) {
  if (!S)
    return nullptr;
  if (auto *CU = dyn_cast<DICompileUnit>(S))
    return CU;
  if (auto *SP = dyn_cast<DISubprogram>(S))
    return SP->getUnit();

  // If it's a namespace stop. namespaces are not associated to any compilation
  // unit.
  if (isa<DINamespace>(S))
    return nullptr;
  return getCUForScope(S->getScope());
}

static void moveGlobalLifetimesIntoGlobalExpressions(Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.retainedNodes");
  if (!NMD)
    return;

  LLVMContext &Context = M.getContext();

  SmallPtrSet<DIGlobalVariable *, 16> Visited;
  DenseMap<DICompileUnit *, SmallVector<Metadata *>> GVExprForCU;
  GVExprForCU.reserve(NMD->getNumOperands());

  for (auto *L : NMD->operands()) {
    auto *Lifetime = cast<DILifetime>(L);
    DIGlobalVariable *GV = cast<DIGlobalVariable>(Lifetime->getObject());
    if (!Visited.insert(GV).second)
      continue;
    DICompileUnit *CU = getCUForScope(GV->getScope());
    if (!CU)
      continue;
    DIExpression *EmptyExpr = DIExpression::get(Context, {});
    GVExprForCU[CU].push_back(
        DIGlobalVariableExpression::get(Context, GV, EmptyExpr));
  }

  for (auto &CUGV : GVExprForCU) {
    CUGV.first->replaceGlobalVariables(MDTuple::get(Context, CUGV.second));
  }

  M.eraseNamedMetadata(NMD);
}

static bool maybeStrip(Module &M, CodeGenOptLevel OptLevel,
                       bool IsNPM = false) {
  if (DisableHeterogeneousDebugVerify || !isHeterogeneousDebug(M))
    return false;
  if (OptLevel == CodeGenOptLevel::None && !IsNPM)
    return false;

  moveGlobalLifetimesIntoGlobalExpressions(M);

  Intrinsic::ID DbgDefKillIntrinsics[] = {Intrinsic::dbg_def,
                                          Intrinsic::dbg_kill};
  for (Intrinsic::ID DbgDefKill : DbgDefKillIntrinsics) {
    Function *Intrinsic = M.getFunction(Intrinsic::getName(DbgDefKill));
    if (!Intrinsic)
      continue;

    while (!Intrinsic->user_empty()) {
      auto *I = cast<DbgDefKillIntrinsic>(Intrinsic->user_back());
      I->eraseFromParent();
    }
    Intrinsic->eraseFromParent();
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
  const CodeGenOptLevel OptLevel;

public:
  static char ID;
  HeterogeneousDebugVerifyLegacy(CodeGenOptLevel OptLevel)
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
llvm::createHeterogeneousDebugVerifyLegacyPass(CodeGenOptLevel OptLevel) {
  return new HeterogeneousDebugVerifyLegacy(OptLevel);
}

namespace llvm {

HeterogeneousDebugVerify::HeterogeneousDebugVerify(CodeGenOptLevel OptLevel)
    : OptLevel(OptLevel) {}

PreservedAnalyses HeterogeneousDebugVerify::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  if (!maybeStrip(M, OptLevel, /*IsNPM=*/true))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // namespace llvm
