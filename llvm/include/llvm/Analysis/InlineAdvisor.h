//===- InlineAdvisor.h - Inlining decision making abstraction -*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_INLINEADVISOR_H_
#define LLVM_INLINEADVISOR_H_

#include <memory>
#include <unordered_set>
#include <vector>

#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class BasicBlock;
class CallBase;
class Function;
class OptimizationRemarkEmitter;

// Default (manual policy) decision making helper APIs. Shared with the legacy
// pass manager inliner.

/// Return the cost only if the inliner should attempt to inline at the given
/// CallSite. If we return the cost, we will emit an optimisation remark later
/// using that cost, so we won't do so from this function. Return None if
/// inlining should not be attempted.
Optional<InlineCost>
shouldInline(CallBase &CB, function_ref<InlineCost(CallBase &CB)> GetInlineCost,
             OptimizationRemarkEmitter &ORE);

/// Emit ORE message.
void emitInlinedInto(OptimizationRemarkEmitter &ORE, DebugLoc DLoc,
                     const BasicBlock *Block, const Function &Callee,
                     const Function &Caller, const InlineCost &IC);

/// Set the inline-remark attribute.
void setInlineRemark(CallBase &CB, StringRef Message);

/// Utility for extracting the inline cost message to a string.
std::string inlineCostStr(const InlineCost &IC);
} // namespace llvm
#endif // LLVM_INLINEADVISOR_H_
