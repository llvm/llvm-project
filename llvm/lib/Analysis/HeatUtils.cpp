//===-- HeatUtils.cpp - Utility for printing heat colors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for printing heat colors based on heuristics or profiling
// information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/HeatUtils.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/Instructions.h"

#include <cmath>

using namespace llvm;

static constexpr unsigned HeatSize = 100;
static constexpr char HeatPalette[HeatSize][8] = {
    "#3d50c3", "#4055c8", "#4358cb", "#465ecf", "#4961d2", "#4c66d6", "#4f69d9",
    "#536edd", "#5572df", "#5977e3", "#5b7ae5", "#5f7fe8", "#6282ea", "#6687ed",
    "#6a8bef", "#6c8ff1", "#7093f3", "#7396f5", "#779af7", "#7a9df8", "#7ea1fa",
    "#81a4fb", "#85a8fc", "#88abfd", "#8caffe", "#8fb1fe", "#93b5fe", "#96b7ff",
    "#9abbff", "#9ebeff", "#a1c0ff", "#a5c3fe", "#a7c5fe", "#abc8fd", "#aec9fc",
    "#b2ccfb", "#b5cdfa", "#b9d0f9", "#bbd1f8", "#bfd3f6", "#c1d4f4", "#c5d6f2",
    "#c7d7f0", "#cbd8ee", "#cedaeb", "#d1dae9", "#d4dbe6", "#d6dce4", "#d9dce1",
    "#dbdcde", "#dedcdb", "#e0dbd8", "#e3d9d3", "#e5d8d1", "#e8d6cc", "#ead5c9",
    "#ecd3c5", "#eed0c0", "#efcebd", "#f1ccb8", "#f2cab5", "#f3c7b1", "#f4c5ad",
    "#f5c1a9", "#f6bfa6", "#f7bca1", "#f7b99e", "#f7b599", "#f7b396", "#f7af91",
    "#f7ac8e", "#f7a889", "#f6a385", "#f5a081", "#f59c7d", "#f4987a", "#f39475",
    "#f29072", "#f08b6e", "#ef886b", "#ed8366", "#ec7f63", "#e97a5f", "#e8765c",
    "#e57058", "#e36c55", "#e16751", "#de614d", "#dc5d4a", "#d85646", "#d65244",
    "#d24b40", "#d0473d", "#cc403a", "#ca3b37", "#c53334", "#c32e31", "#be242e",
    "#bb1b2c", "#b70d28"};

uint64_t llvm::getNumOfCalls(const Function &CallerFunction,
                             const Function &CalledFunction) {
  uint64_t Counter = 0;
  for (const User *U : CalledFunction.users())
    if (auto CI = dyn_cast<CallInst>(U))
      Counter += CI->getCaller() == &CallerFunction;
  return Counter;
}

uint64_t llvm::getMaxFreq(const Function &F, const BlockFrequencyInfo *BFI) {
  uint64_t MaxFreq = 0;
  for (const BasicBlock &BB : F) {
    uint64_t FreqVal = BFI->getBlockFreq(&BB).getFrequency();
    if (FreqVal >= MaxFreq)
      MaxFreq = FreqVal;
  }
  return MaxFreq;
}

std::string llvm::getHeatColor(uint64_t Freq, uint64_t MaxFreq) {
  if (Freq > MaxFreq)
    Freq = MaxFreq;
  double Percent = (Freq > 0) ? log2(double(Freq)) / log2(MaxFreq) : 0;
  return getHeatColor(Percent);
}

std::string llvm::getHeatColor(double Percent) {
  if (Percent > 1.0)
    Percent = 1.0;
  if (Percent < 0.0)
    Percent = 0.0;
  unsigned ColorID = unsigned(round(Percent * (HeatSize - 1.0)));
  return HeatPalette[ColorID];
}
