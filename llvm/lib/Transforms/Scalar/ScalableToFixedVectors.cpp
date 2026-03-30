//===- ScalableToFixedVectors.cpp - Convert scalable to fixed vectors -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts IR operations on scalable vector types to fixed-length
// vectors when the effective length is known and is less than the minimum
// possible scaled vector length. For a scalable vector type with
// element count VF (known min elements), if minvscale * VF > VL, the we can
// convert to a fixed length vector of length VL.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ScalableToFixedVectors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "scalable-to-fixed-vectors"

STATISTIC(
    NumVPStoresEVLLessThanMinWidth,
    "Number of scalable vp.store with constant EVL less than minimum width");

PreservedAnalyses ScalableToFixedVectorsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  // If vscale is unknown, convervatively assume 1
  Attribute VSR = F.getFnAttribute(Attribute::VScaleRange);
  unsigned MinVScale = VSR.isValid() ? VSR.getVScaleRangeMin() : 1;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;

      auto *VPI = dyn_cast<VPIntrinsic>(CI);
      if (!VPI)
        continue;

      if (VPI->getIntrinsicID() != Intrinsic::vp_store)
        continue;

      Value *StoredVal = VPI->getOperand(0);
      Value *EVL = VPI->getOperand(3);

      auto *VTy = dyn_cast<VectorType>(StoredVal->getType());
      if (!VTy)
        continue;

      ElementCount EC = VTy->getElementCount();
      if (!EC.isScalable())
        continue;

      const auto *CInt = dyn_cast<ConstantInt>(EVL);
      if (!CInt)
        continue;

      uint64_t EVLVal = CInt->getZExtValue();
      uint64_t MinElems = EC.getKnownMinValue();
      uint64_t MinWidth = static_cast<uint64_t>(MinVScale) * MinElems;

      if (EVLVal < MinWidth)
        ++NumVPStoresEVLLessThanMinWidth;
    }
  }
  return PreservedAnalyses::all();
}
