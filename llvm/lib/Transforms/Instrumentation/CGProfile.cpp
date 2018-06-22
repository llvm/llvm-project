//===-- CGProfile.cpp -----------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

#include <array>

using namespace llvm;

class CGProfilePass : public ModulePass {
public:
  static char ID;

  CGProfilePass() : ModulePass(ID) {
    initializeCGProfilePassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "CGProfilePass"; }

private:
  bool runOnModule(Module &M) override;
  bool addModuleFlags(
      Module &M,
      MapVector<std::pair<Function *, Function *>, uint64_t> &Counts) const;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
  }
};

bool CGProfilePass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  MapVector<std::pair<Function *, Function *>, uint64_t> Counts;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
    auto &BFI = getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
    for (const auto &BB : F) {
      Optional<uint64_t> BBCount = BFI.getBlockProfileCount(&BB);
      if (!BBCount)
        continue;
      for (const auto &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *CalledF = CI->getCalledFunction();
        if (!CalledF || CalledF->isIntrinsic())
          continue;

        uint64_t &Count = Counts[std::make_pair(&F, CalledF)];
        Count = SaturatingAdd(Count, *BBCount);
      }
    }
  }

  return addModuleFlags(M, Counts);
}

bool CGProfilePass::addModuleFlags(
    Module &M,
    MapVector<std::pair<Function *, Function *>, uint64_t> &Counts) const {
  if (Counts.empty())
    return false;

  LLVMContext &Context = M.getContext();
  MDBuilder MDB(Context);
  std::vector<Metadata *> Nodes;

  for (auto E : Counts) {
    SmallVector<Metadata *, 3> Vals;
    Vals.push_back(ValueAsMetadata::get(E.first.first));
    Vals.push_back(ValueAsMetadata::get(E.first.second));
    Vals.push_back(MDB.createConstant(
        ConstantInt::get(Type::getInt64Ty(Context), E.second)));
    Nodes.push_back(MDNode::get(Context, Vals));
  }

  M.addModuleFlag(Module::Append, "CG Profile", MDNode::get(Context, Nodes));
  return true;
}

char CGProfilePass::ID = 0;
INITIALIZE_PASS_BEGIN(CGProfilePass, "cg-profile",
                      "Generate profile information from the call graph.",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(CGProfilePass, "cg-profile",
                    "Generate profile information from the call graph.", false,
                    false)

ModulePass *llvm::createCGProfilePass() { return new CGProfilePass(); }
