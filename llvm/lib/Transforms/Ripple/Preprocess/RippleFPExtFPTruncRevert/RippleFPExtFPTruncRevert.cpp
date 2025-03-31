//===-- RippleFPExtFPTruncRevert.cpp - Change @llvm.ripple.fpext to fpext -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Ripple/Preprocess/RippleFPExtFPTruncRevert.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace llvm { class Value; }

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "ripple-fpext-fptrunc-revert"

////////////////////////////////////////////////////////////////////////////////
///                     RippleFPExtFPTruncRevertPass                         ///
////////////////////////////////////////////////////////////////////////////////

// Entrypoint for this pass.
PreservedAnalyses
RippleFPExtFPTruncRevertPass::run(Function &F, FunctionAnalysisManager &AM) {
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<TargetLibraryAnalysis>();

  LLVM_DEBUG(dbgs() << "Applying RippleFPExtFPTruncRevertPass pass to '"
                    << F.getName() << "'\n");
  for (auto &I : make_early_inc_range(instructions(F))) {
    Value *Input = nullptr;
    if (match(&I, m_Intrinsic<Intrinsic::ripple_fpext>(m_Value(Input)))) {
      // Found the ripple_fpext intrinsic call
      IRBuilder<> Builder(&I);
      Value *FPExt = Builder.CreateFPExt(Input, I.getType());
      auto IIt = I.getIterator();
      ReplaceInstWithValue(IIt, FPExt);
    } else if (match(&I, m_Intrinsic<Intrinsic::ripple_fptrunc>(
                             m_Value(Input)))) { // Found the ripple_fptrunc
                                                 // intrinsic call
      IRBuilder<> Builder(&I);
      Value *FPTrunc = Builder.CreateFPTrunc(Input, I.getType(), "");
      auto IIt = I.getIterator();
      ReplaceInstWithValue(IIt, FPTrunc);
    }
  }
  return PA;
}
