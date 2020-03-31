//===-- HeatUtils.h - Utility for printing heat colors ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility for printing heat colors based on profiling information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_HEATUTILS_H
#define LLVM_ANALYSIS_HEATUTILS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <string>

namespace llvm {

// Returns the maximum frequency of a BB in a function.
uint64_t getMaxFreq(const Function &F, const BlockFrequencyInfo *BFI);

// Calculates heat color based on current and maximum frequencies.
std::string getHeatColor(uint64_t freq, uint64_t maxFreq);

// Calculates heat color based on percent of "hotness".
std::string getHeatColor(double percent);

} // namespace llvm

#endif
