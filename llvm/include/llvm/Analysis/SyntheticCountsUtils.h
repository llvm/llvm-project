//===- SyntheticCountsUtils.h - utilities for count propagation--*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for synthetic counts propagation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SYNTHETIC_COUNTS_UTILS_H
#define LLVM_ANALYSIS_SYNTHETIC_COUNTS_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/Support/ScaledNumber.h"

namespace llvm {

class CallGraph;
class Function;

using Scaled64 = ScaledNumber<uint64_t>;
void propagateSyntheticCounts(
    const CallGraph &CG, function_ref<Scaled64(CallSite CS)> GetCallSiteRelFreq,
    function_ref<uint64_t(Function *F)> GetCount,
    function_ref<void(Function *F, uint64_t)> AddToCount);
} // namespace llvm

#endif
