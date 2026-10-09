//===- PromoteMemToReg.h - Promote Allocas to Scalars -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to promote alloca instructions to SSA
// registers, by using the SSA construction algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_PROMOTEMEMTOREG_H
#define LLVM_TRANSFORMS_UTILS_PROMOTEMEMTOREG_H

#include "llvm/Support/Compiler.h"
#include <cassert>

namespace llvm {

template <typename T> class ArrayRef;
class AllocaInst;
class DominatorTree;
class AssumptionCache;

/// Either "this alloca is promotable" or the reason it is not, modelled on
/// InlineResult. Converts to bool for convenience.
class AllocaPromotionResult {
  const char *Reason = nullptr;

  AllocaPromotionResult(const char *Reason) : Reason(Reason) {}

public:
  AllocaPromotionResult() = default;

  static AllocaPromotionResult success() { return {}; }

  /// \p Reason is a sentence naming the blocking use, e.g. "Has a volatile
  /// load.", and must outlive the result, so in practice a string literal.
  static AllocaPromotionResult failure(const char *Reason) {
    assert(Reason && "A failure must carry a reason.");
    return AllocaPromotionResult(Reason);
  }

  bool isSuccess() const { return Reason == nullptr; }
  explicit operator bool() const { return isSuccess(); }

  const char *getFailureReason() const {
    assert(!isSuccess() && "Not a failure.");
    return Reason;
  }
};

/// Return whether this alloca is legal for promotion, and if not, why.
///
/// Promotion is legal if there are only loads, stores, and lifetime markers
/// (transitively) using this alloca. This also enforces that there is only
/// ever one layer of bitcasts or GEPs between the alloca and the lifetime
/// markers.
LLVM_ABI AllocaPromotionResult isAllocaPromotable(const AllocaInst *AI);

/// Promote the specified list of alloca instructions into scalar
/// registers, inserting PHI nodes as appropriate.
///
/// This function makes use of DominanceFrontier information.  This function
/// does not modify the CFG of the function at all.  All allocas must be from
/// the same function.
///
LLVM_ABI void PromoteMemToReg(ArrayRef<AllocaInst *> Allocas, DominatorTree &DT,
                              AssumptionCache *AC = nullptr);

} // End llvm namespace

#endif
