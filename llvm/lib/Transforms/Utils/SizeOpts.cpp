//===-- SizeOpts.cpp - code size optimization related code ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some shared code size optimization related code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
using namespace llvm;

static cl::opt<bool> ProfileGuidedSizeOpt(
    "pgso", cl::Hidden, cl::init(true),
    cl::desc("Enable the profile guided size optimization. "));

bool llvm::shouldOptimizeForSize(Function *F, ProfileSummaryInfo *PSI,
                                 BlockFrequencyInfo *BFI) {
  assert(F);
  if (!PSI || !BFI || !PSI->hasProfileSummary())
    return false;
  return ProfileGuidedSizeOpt && PSI->isFunctionColdInCallGraph(F, *BFI);
}

bool llvm::shouldOptimizeForSize(BasicBlock *BB, ProfileSummaryInfo *PSI,
                                 BlockFrequencyInfo *BFI) {
  assert(BB);
  if (!PSI || !BFI || !PSI->hasProfileSummary())
    return false;
  return ProfileGuidedSizeOpt && PSI->isColdBlock(BB, BFI);
}
