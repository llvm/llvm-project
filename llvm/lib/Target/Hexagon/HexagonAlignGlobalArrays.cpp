//===- HexagonAlignGlobalArrays.cpp - Align Global Arrays -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass increases the alignment of global integer arrays (char, short,
// int), including multi-dimensional arrays, to an 8-byte boundary. This is
// done to make their alignment compatible with GCC. The pass is disabled by
// default and can be enabled with -hexagon-align-global-arrays.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-global-array-alignment"

static cl::opt<bool> EnableGlobalArrayAlignment(
    "hexagon-align-global-arrays",
    cl::desc("Align global integer arrays to an 8-byte boundary"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> DisableHexAlignOptByteHalf(
    "hexagon-disable-align-opt-byte-half",
    cl::desc("Disable optimizing byte and half-word array alignment when "
             "optimizing for size"),
    cl::Hidden);

namespace {

class HexagonAlignGlobalArrays : public ModulePass {
  bool OptForSize;

public:
  static char ID;

  explicit HexagonAlignGlobalArrays(bool Os = false)
      : ModulePass(ID), OptForSize(Os) {}

  StringRef getPassName() const override {
    return "Hexagon Global Array Alignment";
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

char HexagonAlignGlobalArrays::ID = 0;

INITIALIZE_PASS(HexagonAlignGlobalArrays, "hexagon-global-array-alignment",
                "Align Global Arrays to 8-byte", false, false)

ModulePass *llvm::createHexagonAlignGlobalArrays(bool Os) {
  return new HexagonAlignGlobalArrays(Os);
}

// Get the underlying element type of an array. This is useful if the array is
// multi-dimensional.
static Type *getUnderlyingArrayElmTy(Type *Ty) {
  // Ty is guaranteed to be an array type.
  Type *ElTy = cast<ArrayType>(Ty)->getElementType();
  if (ElTy->isArrayTy())
    ElTy = getUnderlyingArrayElmTy(ElTy);
  return ElTy;
}

bool HexagonAlignGlobalArrays::runOnModule(Module &M) {
  if (!EnableGlobalArrayAlignment)
    return false;

  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  for (GlobalVariable &GV : M.globals()) {
    Type *VT = GV.getValueType();

    // Compute the current alignment, falling back to the ABI alignment.
    MaybeAlign GVAlign = GV.getAlign();
    if (!GVAlign && VT->isSized())
      GVAlign = DL.getABITypeAlign(VT);

    // Align global integer arrays (char, short, int) to an 8-byte boundary.
    // This makes their alignment compatible with GCC. Chars and shorts keep
    // their native alignment if not explicitly aligned to a larger size.
    if (VT->isArrayTy()) {
      Type *ElTy = getUnderlyingArrayElmTy(VT);
      if (ElTy->isIntegerTy()) {
        if (OptForSize && GVAlign && *GVAlign <= Align(2) &&
            !DisableHexAlignOptByteHalf) {
          ; // Do nothing.
        } else {
          MaybeAlign NewAlign = std::max(GVAlign.valueOrOne(), Align(8));
          if (NewAlign != GVAlign) {
            GV.setAlignment(NewAlign);
            Changed = true;
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << GV << '\n');
  }

  return Changed;
}
