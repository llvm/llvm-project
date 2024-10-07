//===- BottomUpVec.cpp - A bottom-up vectorizer pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/NullPass.h"

namespace llvm::sandboxir {

cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

/// A magic string for the default pass pipeline.
const char *DefaultPipelineMagicStr = "*";

cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of vectorizer passes. If not set "
             "we run the predefined pipeline."));

static void registerAllRegionPasses(sandboxir::PassRegistry &PR) {
  PR.registerPass(std::make_unique<sandboxir::NullPass>());
}

static sandboxir::RegionPassManager &
parseAndCreatePassPipeline(sandboxir::PassRegistry &PR, StringRef Pipeline) {
  static constexpr const char EndToken = '\0';
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = std::string(Pipeline) + EndToken;
  int FlagBeginIdx = 0;
  auto &RPM = static_cast<sandboxir::RegionPassManager &>(
      PR.registerPass(std::make_unique<sandboxir::RegionPassManager>("rpm")));

  for (auto [Idx, C] : enumerate(PipelineStr)) {
    // Keep moving Idx until we find the end of the pass name.
    bool FoundDelim = C == EndToken || C == PR.PassDelimToken;
    if (!FoundDelim)
      continue;
    unsigned Sz = Idx - FlagBeginIdx;
    std::string PassName(&PipelineStr[FlagBeginIdx], Sz);
    FlagBeginIdx = Idx + 1;

    // Get the pass that corresponds to PassName and add it to the pass manager.
    auto *Pass = PR.getPassByName(PassName);
    if (Pass == nullptr) {
      errs() << "Pass '" << PassName << "' not registered!\n";
      exit(1);
    }
    // TODO: Add a type check here. The downcast is correct as long as
    // registerAllRegionPasses only registers regions passes.
    RPM.addPass(static_cast<sandboxir::RegionPass *>(Pass));
  }
  return RPM;
}

BottomUpVec::BottomUpVec() : FunctionPass("bottom-up-vec") {
  registerAllRegionPasses(PR);

  // Create a pipeline to be run on each Region created by BottomUpVec.
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // Create the default pass pipeline.
    RPM = &static_cast<sandboxir::RegionPassManager &>(PR.registerPass(
        std::make_unique<sandboxir::FunctionPassManager>("rpm")));
    // TODO: Add passes to the default pipeline.
  } else {
    // Create the user-defined pipeline.
    RPM = &parseAndCreatePassPipeline(PR, UserDefinedPassPipeline);
  }
}

// TODO: This is a temporary function that returns some seeds.
//       Replace this with SeedCollector's function when it lands.
static llvm::SmallVector<Value *, 4> collectSeeds(BasicBlock &BB) {
  llvm::SmallVector<Value *, 4> Seeds;
  for (auto &I : BB)
    if (auto *SI = llvm::dyn_cast<StoreInst>(&I))
      Seeds.push_back(SI);
  return Seeds;
}

static SmallVector<Value *, 4> getOperand(ArrayRef<Value *> Bndl,
                                          unsigned OpIdx) {
  SmallVector<Value *, 4> Operands;
  for (Value *BndlV : Bndl) {
    auto *BndlI = cast<Instruction>(BndlV);
    Operands.push_back(BndlI->getOperand(OpIdx));
  }
  return Operands;
}

void BottomUpVec::vectorizeRec(ArrayRef<Value *> Bndl) {
  auto LegalityRes = Legality.canVectorize(Bndl);
  switch (LegalityRes.getSubclassID()) {
  case LegalityResultID::Widen: {
    auto *I = cast<Instruction>(Bndl[0]);
    for (auto OpIdx : seq<unsigned>(I->getNumOperands())) {
      auto OperandBndl = getOperand(Bndl, OpIdx);
      vectorizeRec(OperandBndl);
    }
    break;
  }
  }
}

void BottomUpVec::tryVectorize(ArrayRef<Value *> Bndl) { vectorizeRec(Bndl); }

bool BottomUpVec::runOnFunction(Function &F) {
  if (PrintPassPipeline) {
    RPM->printPipeline(outs());
    return false;
  }

  Change = false;
  // TODO: Start from innermost BBs first
  for (auto &BB : F) {
    // TODO: Replace with proper SeedCollector function.
    auto Seeds = collectSeeds(BB);
    // TODO: Slice Seeds into smaller chunks.
    // TODO: If vectorization succeeds, run the RegionPassManager on the
    // resulting region.
    if (Seeds.size() >= 2)
      tryVectorize(Seeds);
  }
  return Change;
}

} // namespace llvm::sandboxir
