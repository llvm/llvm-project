//===- LoopRotation.cpp - Loop Rotation Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Rotation Pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "ScalarOptions.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/LoopRotationUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "loop-rotate"

LoopRotatePass::LoopRotatePass(bool EnableHeaderDuplication, bool PrepareForLTO,
                               bool CheckExitCount)
    : EnableHeaderDuplication(EnableHeaderDuplication),
      PrepareForLTO(PrepareForLTO), CheckExitCount(CheckExitCount) {}

void LoopRotatePass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LoopRotatePass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";
  if (!EnableHeaderDuplication)
    OS << "no-";
  OS << "header-duplication;";

  if (!PrepareForLTO)
    OS << "no-";
  OS << "prepare-for-lto;";

  if (!CheckExitCount)
    OS << "no-";
  OS << "check-exit-count";
  OS << ">";
}

PreservedAnalyses LoopRotatePass::run(Loop &L, LoopAnalysisManager &AM,
                                      LoopStandardAnalysisResults &AR,
                                      LPMUpdater &) {
  const ScalarOptions &Opts = ScalarOptions::Global;
  // Vectorization requires loop-rotation. Use default threshold for loops the
  // user explicitly marked for vectorization, even when header duplication is
  // disabled.
  int Threshold = EnableHeaderDuplication &&
                          (!L.getHeader()->getParent()->hasMinSize() ||
                           Opts.enable_loop_header_duplication_at_minsize ||
                           hasVectorizeTransformation(&L) == TM_ForcedByUser)
                      ? Opts.rotation_max_header_size
                      : 0;
  const DataLayout &DL = L.getHeader()->getDataLayout();
  const SimplifyQuery SQ = getBestSimplifyQuery(AR, DL);

  std::optional<MemorySSAUpdater> MSSAU;
  if (AR.MSSA)
    MSSAU = MemorySSAUpdater(AR.MSSA);
  bool Changed = LoopRotation(
      &L, &AR.LI, &AR.TTI, &AR.AC, &AR.DT, &AR.SE, MSSAU ? &*MSSAU : nullptr,
      SQ, false, Threshold, false,
      PrepareForLTO || Opts.rotation_prepare_for_lto, CheckExitCount);

  if (!Changed)
    return PreservedAnalyses::all();

  if (AR.MSSA && VerifyMemorySSA)
    AR.MSSA->verifyMemorySSA();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}
