//===- BitcodeWriterPass.cpp - Bitcode writing pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BitcodeWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
using namespace llvm;

PreservedAnalyses BitcodeWriterPass::run(Module &M, ModuleAnalysisManager &AM) {
  M.removeDebugIntrinsicDeclarations();

  ProfileSummaryInfo &PSI = AM.getResult<ProfileSummaryAnalysis>(M);
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool NeedSSI = needsParamAccessSummary(M);
  std::unique_ptr<ModuleSummaryIndex> NewIndex = nullptr;
  if (EmitSummaryIndex) {
    NewIndex = std::make_unique<ModuleSummaryIndex>(buildModuleSummaryIndex(
        M,
        [&FAM](const Function &F) {
          return &FAM.getResult<BlockFrequencyAnalysis>(
              *const_cast<Function *>(&F));
        },
        &PSI, TM,
        [&FAM, NeedSSI](const Function &F) -> const StackSafetyInfo * {
          return NeedSSI ? &FAM.getResult<StackSafetyAnalysis>(
                               const_cast<Function &>(F))
                         : nullptr;
        }));
  }
  WriteBitcodeToFile(M, OS, ShouldPreserveUseListOrder, NewIndex.get(),
                     EmitModuleHash, /*ModHash=*/nullptr, TM);
  return PreservedAnalyses::all();
}

namespace {
  class WriteBitcodePass : public ModulePass {
    raw_ostream &OS; // raw_ostream to print on
    bool ShouldPreserveUseListOrder;
    const TargetMachine *TM;

  public:
    static char ID; // Pass identification, replacement for typeid
    WriteBitcodePass() : ModulePass(ID), OS(dbgs()) {
      initializeWriteBitcodePassPass(*PassRegistry::getPassRegistry());
    }

    explicit WriteBitcodePass(raw_ostream &o, bool ShouldPreserveUseListOrder,
                              const TargetMachine *TM = nullptr)
        : ModulePass(ID), OS(o),
          ShouldPreserveUseListOrder(ShouldPreserveUseListOrder), TM(TM) {
      initializeWriteBitcodePassPass(*PassRegistry::getPassRegistry());
    }

    StringRef getPassName() const override { return "Bitcode Writer"; }

    bool runOnModule(Module &M) override {
      M.removeDebugIntrinsicDeclarations();

      WriteBitcodeToFile(M, OS, ShouldPreserveUseListOrder, /*Index=*/nullptr,
                         /*GenerateHash=*/false, /*ModHash=*/nullptr, TM);

      return false;
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char WriteBitcodePass::ID = 0;
INITIALIZE_PASS_BEGIN(WriteBitcodePass, "write-bitcode", "Write Bitcode", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(ModuleSummaryIndexWrapperPass)
INITIALIZE_PASS_END(WriteBitcodePass, "write-bitcode", "Write Bitcode", false,
                    true)

ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str,
                                          bool ShouldPreserveUseListOrder,
                                          const TargetMachine *TM) {
  return new WriteBitcodePass(Str, ShouldPreserveUseListOrder, TM);
}

bool llvm::isBitcodeWriterPass(Pass *P) {
  return P->getPassID() == (llvm::AnalysisID)&WriteBitcodePass::ID;
}
