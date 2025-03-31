//===--- RippleFPExtFPTrunc.cpp - Update fpext inst to @llvm.ripple.fpe ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Ripple/Preprocess/RippleFPExtFPTrunc.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "ripple-fpext-fptrunc"

////////////////////////////////////////////////////////////////////////////////
///                          RippleFPExtFPTrunc                              ///
////////////////////////////////////////////////////////////////////////////////

// Entrypoint for this pass.
PreservedAnalyses RippleFPExtFPTruncPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<TargetLibraryAnalysis>();

  LLVM_DEBUG(dbgs() << "Applying RippleFPExtFPTruncPass to '" << F.getName()
                    << "'\n");

  for (auto &I : make_early_inc_range(instructions(F))) {
    if (auto *FPExt = dyn_cast<FPExtInst>(&I)) {
      if (FPExt->getOperand(0)->getType()->getScalarType()->isBFloatTy()) {
        IRBuilder<> Builder(FPExt);
        auto *NewInst =
            Builder.CreateIntrinsic(I.getType(), Intrinsic::ripple_fpext,
                                    {FPExt->getOperand(0)}, FMFSource(FPExt));
        auto IIt = I.getIterator();
        ReplaceInstWithValue(IIt, NewInst);
      }
    }
    if (auto *FPTrunc = dyn_cast<FPTruncInst>(&I)) {
      if (FPTrunc->getType()->getScalarType()->isBFloatTy()) {
        IRBuilder<> Builder(FPTrunc);
        auto *NewInst = Builder.CreateIntrinsic(
            I.getType(), Intrinsic::ripple_fptrunc, {FPTrunc->getOperand(0)},
            FMFSource(FPTrunc));
        auto IIt = I.getIterator();
        ReplaceInstWithValue(IIt, NewInst);
      }
    }
  }

  return PA;
}
