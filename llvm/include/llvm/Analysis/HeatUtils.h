//===-- HeatUtils.h - Utility for printing heat colors ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for printing heat colors based on profiling information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_HEATUTILS_H
#define LLVM_ANALYSIS_HEATUTILS_H

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <string>

namespace llvm {

class BlockFrequencyInfo;
class Function;

// Returns number of calls of calledFunction by callerFunction.
LLVM_ABI uint64_t getNumOfCalls(const Function &CallerFunction,
                                const Function &CalledFunction);

// Returns the maximum frequency of a BB in a function.
LLVM_ABI uint64_t getMaxFreq(const Function &F, const BlockFrequencyInfo *BFI);

// Calculates heat color based on current and maximum frequencies.
LLVM_ABI std::string getHeatColor(uint64_t Freq, uint64_t MaxFreq);

// Calculates heat color based on percent of "hotness".
LLVM_ABI std::string getHeatColor(double Percent);

} // namespace llvm

#endif
