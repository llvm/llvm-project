//===-- NVPTXIncreaseAlignment.cpp - Increase alignment for local arrays --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple pass that looks at local memory arrays that are statically
// sized and potentially increases their alignment. This enables vectorization
// of loads/stores to these arrays if not explicitly specified by the client.
//
// TODO: Ideally we should do a bin-packing of local arrays to maximize
// alignments while minimizing holes.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static cl::opt<bool>
    MaxLocalArrayAlignment("nvptx-use-max-local-array-alignment",
                           cl::init(false), cl::Hidden,
                           cl::desc("Use maximum alignment for local memory"));

static Align getMaxLocalArrayAlignment(const TargetTransformInfo &TTI) {
  const unsigned MaxBitWidth =
      TTI.getLoadStoreVecRegBitWidth(NVPTXAS::ADDRESS_SPACE_LOCAL);
  return Align(MaxBitWidth / 8);
}

namespace {
struct NVPTXIncreaseLocalAlignment {
  const Align MaxAlign;

  NVPTXIncreaseLocalAlignment(const TargetTransformInfo &TTI)
      : MaxAlign(getMaxLocalArrayAlignment(TTI)) {}

  bool run(Function &F);
  bool updateAllocaAlignment(AllocaInst *Alloca, const DataLayout &DL);
  Align getAggressiveArrayAlignment(unsigned ArraySize);
  Align getConservativeArrayAlignment(unsigned ArraySize);
};
} // namespace

/// Get the maximum useful alignment for an array. This is more likely to
/// produce holes in the local memory.
///
/// Choose an alignment large enough that the entire array could be loaded with
/// a single vector load (if possible). Cap the alignment at
/// MaxPTXArrayAlignment.
Align NVPTXIncreaseLocalAlignment::getAggressiveArrayAlignment(
    const unsigned ArraySize) {
  return std::min(MaxAlign, Align(PowerOf2Ceil(ArraySize)));
}

/// Get the alignment of arrays that reduces the chances of leaving holes when
/// arrays are allocated within a contiguous memory buffer (like shared memory
/// and stack). Holes are still possible before and after the array allocation.
///
/// Choose the largest alignment such that the array size is a multiple of the
/// alignment. If all elements of the buffer are allocated in order of
/// alignment (higher to lower) no holes will be left.
Align NVPTXIncreaseLocalAlignment::getConservativeArrayAlignment(
    const unsigned ArraySize) {
  return commonAlignment(MaxAlign, ArraySize);
}

/// Find a better alignment for local arrays
bool NVPTXIncreaseLocalAlignment::updateAllocaAlignment(AllocaInst *Alloca,
                                                        const DataLayout &DL) {
  // Looking for statically sized local arrays
  if (!Alloca->isStaticAlloca())
    return false;

  const auto ArraySize = Alloca->getAllocationSize(DL);
  if (!(ArraySize && ArraySize->isFixed()))
    return false;

  const auto ArraySizeValue = ArraySize->getFixedValue();
  const Align PreferredAlignment =
      MaxLocalArrayAlignment ? getAggressiveArrayAlignment(ArraySizeValue)
                             : getConservativeArrayAlignment(ArraySizeValue);

  if (PreferredAlignment > Alloca->getAlign()) {
    Alloca->setAlignment(PreferredAlignment);
    return true;
  }

  return false;
}

bool NVPTXIncreaseLocalAlignment::run(Function &F) {
  bool Changed = false;
  const auto &DL = F.getParent()->getDataLayout();

  BasicBlock &EntryBB = F.getEntryBlock();
  for (Instruction &I : EntryBB)
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I))
      Changed |= updateAllocaAlignment(Alloca, DL);

  return Changed;
}

namespace {
struct NVPTXIncreaseLocalAlignmentLegacyPass : public FunctionPass {
  static char ID;
  NVPTXIncreaseLocalAlignmentLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
  StringRef getPassName() const override {
    return "NVPTX Increase Local Alignment";
  }
};
} // namespace

char NVPTXIncreaseLocalAlignmentLegacyPass::ID = 0;
INITIALIZE_PASS(NVPTXIncreaseLocalAlignmentLegacyPass,
                "nvptx-increase-local-alignment",
                "Increase alignment for statically sized alloca arrays", false,
                false)

FunctionPass *llvm::createNVPTXIncreaseLocalAlignmentPass() {
  return new NVPTXIncreaseLocalAlignmentLegacyPass();
}

bool NVPTXIncreaseLocalAlignmentLegacyPass::runOnFunction(Function &F) {
  const auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  return NVPTXIncreaseLocalAlignment(TTI).run(F);
}

PreservedAnalyses
NVPTXIncreaseLocalAlignmentPass::run(Function &F,
                                     FunctionAnalysisManager &FAM) {
  const auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  bool Changed = NVPTXIncreaseLocalAlignment(TTI).run(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
