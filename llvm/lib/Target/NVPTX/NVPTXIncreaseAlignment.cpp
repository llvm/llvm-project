//===-- NVPTXIncreaseAlignment.cpp - Increase alignment for local arrays --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple pass that looks at local memory allocas that are statically
// sized and potentially increases their alignment. This enables vectorization
// of loads/stores to these allocas if not explicitly specified by the client.
//
// TODO: Ideally we should do a bin-packing of local allocas to maximize
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

static cl::opt<unsigned> MinLocalArrayAlignment(
    "nvptx-ensure-minimum-local-alignment", cl::init(16), cl::Hidden,
    cl::desc(
        "Ensure local memory objects are at least this aligned (default 16)"));

static Align getMaxLocalArrayAlignment(const TargetTransformInfo &TTI) {
  const unsigned MaxBitWidth =
      TTI.getLoadStoreVecRegBitWidth(NVPTXAS::ADDRESS_SPACE_LOCAL);
  return Align(MaxBitWidth / 8);
}

namespace {
struct NVPTXIncreaseLocalAlignment {
  const Align MaxUsableAlign;

  NVPTXIncreaseLocalAlignment(const TargetTransformInfo &TTI)
      : MaxUsableAlign(getMaxLocalArrayAlignment(TTI)) {}

  bool run(Function &F);
  bool updateAllocaAlignment(AllocaInst *Alloca, const DataLayout &DL);
  Align getMaxUsefulArrayAlignment(unsigned ArraySize);
  Align getMaxSafeLocalAlignment(unsigned ArraySize);
};
} // namespace

/// Get the maximum useful alignment for an allocation. This is more likely to
/// produce holes in the local memory.
///
/// Choose an alignment large enough that the entire alloca could be loaded
/// with a single vector load (if possible). Cap the alignment at
/// MinLocalArrayAlignment and MaxUsableAlign.
Align NVPTXIncreaseLocalAlignment::getMaxUsefulArrayAlignment(
    const unsigned ArraySize) {
  const Align UpperLimit =
      std::min(MaxUsableAlign, Align(MinLocalArrayAlignment));
  return std::min(UpperLimit, Align(PowerOf2Ceil(ArraySize)));
}

/// Get the alignment of allocas that reduces the chances of leaving holes when
/// they are allocated within a contiguous memory buffer (like the stack).
/// Holes are still possible before and after the allocation.
///
/// Choose the largest alignment such that the allocation size is a multiple of
/// the alignment. If all elements of the buffer are allocated in order of
/// alignment (higher to lower) no holes will be left.
Align NVPTXIncreaseLocalAlignment::getMaxSafeLocalAlignment(
    const unsigned ArraySize) {
  return commonAlignment(MaxUsableAlign, ArraySize);
}

/// Find a better alignment for local allocas.
bool NVPTXIncreaseLocalAlignment::updateAllocaAlignment(AllocaInst *Alloca,
                                                        const DataLayout &DL) {
  if (!Alloca->isStaticAlloca())
    return false;

  const auto ArraySize = Alloca->getAllocationSize(DL);
  if (!(ArraySize && ArraySize->isFixed()))
    return false;

  const auto ArraySizeValue = ArraySize->getFixedValue();
  if (ArraySizeValue == 0)
    return false;

  const Align NewAlignment =
      std::max(getMaxSafeLocalAlignment(ArraySizeValue),
               getMaxUsefulArrayAlignment(ArraySizeValue));

  if (NewAlignment > Alloca->getAlign()) {
    Alloca->setAlignment(NewAlignment);
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
                "Increase alignment for statically sized allocas", false, false)

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
