//===- HexagonAlignGlobalArrays.cpp - Align Global Arrays -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass increases the alignment of global integer arrays (char, short,
// int), including multi-dimensional arrays, to an 8-byte boundary. This gives
// their base address a wider alignment, which is beneficial for the wide
// (double-word) loads and stores available on Hexagon.
//
// When optimizing to reduce .rodata size, byte and half-word arrays already at
// an alignment of two bytes or less are left at their natural alignment; only
// word arrays are promoted. This size behavior can be turned off with
// -hexagon-disable-align-opt-byte-half.
//
// The pass is enabled by default and can be disabled with
// -hexagon-disable-global-array-align.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-global-array-alignment"

static cl::opt<bool> DisableGlobalArrayAlignment(
    "hexagon-disable-global-array-align",
    cl::desc("Disable aligning global integer arrays to an 8-byte boundary"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> DisableHexAlignOptByteHalf(
    "hexagon-disable-align-opt-byte-half",
    cl::desc("Disable keeping byte and half-word arrays at their natural "
             "alignment when reducing .rodata size"),
    cl::Hidden);

namespace {

class HexagonAlignGlobalArrays : public ModulePass {
  bool ReduceRodataSize;

public:
  static char ID;

  explicit HexagonAlignGlobalArrays(bool ReduceRodataSize = false)
      : ModulePass(ID), ReduceRodataSize(ReduceRodataSize) {}

  StringRef getPassName() const override {
    return "Hexagon Global Array Alignment";
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

char HexagonAlignGlobalArrays::ID = 0;

INITIALIZE_PASS(HexagonAlignGlobalArrays, "hexagon-global-array-alignment",
                "Align Global Arrays to 8-byte", false, false)

ModulePass *llvm::createHexagonAlignGlobalArrays(bool ReduceRodataSize) {
  return new HexagonAlignGlobalArrays(ReduceRodataSize);
}

// Get the underlying element type of an array. This is useful if the array is
// multi-dimensional.
static Type *getUnderlyingArrayElmTy(Type *Ty) {
  // Ty is guaranteed to be an array type.
  Type *ElTy = cast<ArrayType>(Ty)->getElementType();
  while (ElTy->isArrayTy())
    ElTy = cast<ArrayType>(ElTy)->getElementType();
  return ElTy;
}

bool HexagonAlignGlobalArrays::runOnModule(Module &M) {
  if (DisableGlobalArrayAlignment)
    return false;

  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  for (GlobalVariable &GV : M.globals()) {
    Type *VT = GV.getValueType();
    if (!VT->isArrayTy())
      continue;

    Type *ElTy = getUnderlyingArrayElmTy(VT);
    if (!ElTy->isIntegerTy())
      continue;

    // Skip globals whose alignment cannot be safely raised, e.g. declarations,
    // weak/interposable definitions, and section-pinned globals.
    if (!GV.canIncreaseAlignment())
      continue;

    // Compute the current alignment, falling back to the ABI alignment.
    MaybeAlign GVAlign = GV.getAlign();
    if (!GVAlign && VT->isSized())
      GVAlign = DL.getABITypeAlign(VT);

    // Align integer arrays to an 8-byte boundary. When reducing .rodata size,
    // leave byte and half-word arrays that are already at an alignment of two
    // bytes or less at their natural alignment; word arrays are still promoted.
    if (!ReduceRodataSize || !GVAlign || *GVAlign > Align(2) ||
        DisableHexAlignOptByteHalf) {
      MaybeAlign NewAlign = std::max(GVAlign.valueOrOne(), Align(8));
      if (NewAlign != GVAlign) {
        GV.setAlignment(NewAlign);
        Changed = true;
        LLVM_DEBUG(dbgs() << GV << '\n');
      }
    }
  }

  return Changed;
}
