//===-- NVPTXIncreaseAlignment.cpp - Increase alignment for local arrays --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple pass that looks at local memory arrays that are statically
// sized and sets an appropriate alignment for them. This enables vectorization
// of loads/stores to these arrays if not explicitly specified by the client.
//
// TODO: Ideally we should do a bin-packing of local arrays to maximize
// alignments while minimizing holes.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

static cl::opt<bool>
    MaxLocalArrayAlignment("nvptx-use-max-local-array-alignment",
                           cl::init(false), cl::Hidden,
                           cl::desc("Use maximum alignment for local memory"));

static constexpr Align MaxPTXArrayAlignment = Align::Constant<16>();

/// Get the maximum useful alignment for an array. This is more likely to
/// produce holes in the local memory.
///
/// Choose an alignment large enough that the entire array could be loaded with
/// a single vector load (if possible). Cap the alignment at
/// MaxPTXArrayAlignment.
static Align getAggressiveArrayAlignment(const unsigned ArraySize) {
  return std::min(MaxPTXArrayAlignment, Align(PowerOf2Ceil(ArraySize)));
}

/// Get the alignment of arrays that reduces the chances of leaving holes when
/// arrays are allocated within a contiguous memory buffer (like shared memory
/// and stack). Holes are still possible before and after the array allocation.
///
/// Choose the largest alignment such that the array size is a multiple of the
/// alignment. If all elements of the buffer are allocated in order of
/// alignment (higher to lower) no holes will be left.
static Align getConservativeArrayAlignment(const unsigned ArraySize) {
  return commonAlignment(MaxPTXArrayAlignment, ArraySize);
}

/// Find a better alignment for local arrays
static bool updateAllocaAlignment(const DataLayout &DL, AllocaInst *Alloca) {
  // Looking for statically sized local arrays
  if (!Alloca->isStaticAlloca())
    return false;

  // For now, we only support array allocas
  if (!(Alloca->isArrayAllocation() || Alloca->getAllocatedType()->isArrayTy()))
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

static bool runSetLocalArrayAlignment(Function &F) {
  bool Changed = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  BasicBlock &EntryBB = F.getEntryBlock();
  for (Instruction &I : EntryBB)
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I))
      Changed |= updateAllocaAlignment(DL, Alloca);

  return Changed;
}

namespace {
struct NVPTXIncreaseLocalAlignmentLegacyPass : public FunctionPass {
  static char ID;
  NVPTXIncreaseLocalAlignmentLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
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
  return runSetLocalArrayAlignment(F);
}

PreservedAnalyses
NVPTXIncreaseLocalAlignmentPass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = runSetLocalArrayAlignment(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
