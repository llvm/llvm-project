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

static cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

/// A magic string for the default pass pipeline.
static const char *DefaultPipelineMagicStr = "*";

static cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of vectorizer passes. If not set "
             "we run the predefined pipeline."));

static std::unique_ptr<RegionPass> createRegionPass(StringRef Name) {
#define REGION_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME)                                                            \
    return std::make_unique<decltype(CREATE_PASS)>(CREATE_PASS);
#include "PassRegistry.def"
  return nullptr;
}

/// Adds to `RPM` a sequence of region passes described by `Pipeline`.
static void parseAndCreatePassPipeline(RegionPassManager &RPM,
                                       StringRef Pipeline) {
  static constexpr const char EndToken = '\0';
  static constexpr const char PassDelimToken = ',';
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = std::string(Pipeline) + EndToken;
  int FlagBeginIdx = 0;

  for (auto [Idx, C] : enumerate(PipelineStr)) {
    // Keep moving Idx until we find the end of the pass name.
    bool FoundDelim = C == EndToken || C == PassDelimToken;
    if (!FoundDelim)
      continue;
    unsigned Sz = Idx - FlagBeginIdx;
    std::string PassName(&PipelineStr[FlagBeginIdx], Sz);
    FlagBeginIdx = Idx + 1;

    // Get the pass that corresponds to PassName and add it to the pass manager.
    auto RegionPass = createRegionPass(PassName);
    if (RegionPass == nullptr) {
      errs() << "Pass '" << PassName << "' not registered!\n";
      exit(1);
    }
    RPM.addPass(std::move(RegionPass));
  }
}

BottomUpVec::BottomUpVec() : FunctionPass("bottom-up-vec"), RPM("rpm") {
  // Create a pipeline to be run on each Region created by BottomUpVec.
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // TODO: Add default passes to RPM.
  } else {
    // Create the user-defined pipeline.
    parseAndCreatePassPipeline(RPM, UserDefinedPassPipeline);
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
    RPM.printPipeline(outs());
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
