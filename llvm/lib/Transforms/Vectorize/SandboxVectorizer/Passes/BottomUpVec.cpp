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

BottomUpVec::BottomUpVec() : FunctionPass("bottom-up-vec"), RPM("rpm") {
  // Create a pipeline to be run on each Region created by BottomUpVec.
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // TODO: Add default passes to RPM.
  } else {
    // Create the user-defined pipeline.
    RPM.setPassPipeline(UserDefinedPassPipeline, createRegionPass);
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
