//===- LoopVectorizationUtils.h - Utilities for LoopVectorize -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides declaration for stateless functions that are used by the
/// LoopVectorize and its related files.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONUTILS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace LoopVectorizationUtils {

// Loop vectorization hints how the epilogue/tail loop should be
// lowered.
enum EpilogueLowering {

  // The default: allowing epilogues.
  EpilogueAllowed,

  // Vectorization with OptForSize: don't allow epilogues.
  EpilogueNotAllowedOptSize,

  // A special case of vectorisation with OptForSize: loops with a very small
  // trip count are considered for vectorization under OptForSize, thereby
  // making sure the cost of their loop body is dominant, free of runtime
  // guards and scalar iteration overheads.
  EpilogueNotAllowedLowTripLoop,

  // Loop hint indicating an epilogue is undesired, apply tail folding.
  EpilogueNotNeededFoldTail,

  // Directive indicating we must either fold the epilogue/tail or not vectorize
  EpilogueNotAllowedFoldTail
};

/// Reports a vectorization failure: print \p DebugMsg for debugging
/// purposes along with the corresponding optimization remark \p RemarkName.
/// If \p I is passed, it is an instruction that prevents vectorization.
/// Otherwise, the loop \p TheLoop is used for the location of the remark.
void reportVectorizationFailure(const char *PassName, const StringRef DebugMsg,
                                const StringRef OREMsg, const StringRef ORETag,
                                OptimizationRemarkEmitter *ORE,
                                const Loop *TheLoop, Instruction *I = nullptr);

/// Same as above, but the debug message and optimization remark are identical
inline void reportVectorizationFailure(const char *PassName,
                                       const StringRef DebugMsg,
                                       const StringRef ORETag,
                                       OptimizationRemarkEmitter *ORE,
                                       const Loop *TheLoop,
                                       Instruction *I = nullptr) {
  reportVectorizationFailure(PassName, DebugMsg, DebugMsg, ORETag, ORE, TheLoop,
                             I);
}

/// Reports an informative message: print \p Msg for debugging purposes as well
/// as an optimization remark. Uses either \p I as location of the remark, or
/// otherwise \p TheLoop. If \p DL is passed, use it as debug location for the
/// remark.
void reportVectorizationInfo(const char *PassName, const StringRef Msg,
                             const StringRef ORETag,
                             OptimizationRemarkEmitter *ORE,
                             const Loop *TheLoop, Instruction *I = nullptr,
                             DebugLoc DL = {});

/// Report successful vectorization of the loop. In case an outer loop is
/// vectorized, prepend "outer" to the vectorization remark.
void reportVectorization(const char *PassName, OptimizationRemarkEmitter *ORE,
                         Loop *TheLoop, ElementCount VFWidth, unsigned IC);

/// A version of ScalarEvolution::getSmallConstantTripCount that returns an
/// ElementCount to include loops whose trip count is a function of vscale.
ElementCount getSmallConstantTripCount(ScalarEvolution *SE, const Loop *L);

/// Get the maximum trip count for \p L from the SCEV unsigned range, excluding
/// zero from the range. Only valid when not folding the tail, as the minimum
/// iteration count check guards against a zero trip count. Returns 0 if
/// unknown.
unsigned getMaxTCFromNonZeroRange(PredicatedScalarEvolution &PSE,
                                  const Loop *L);
} // namespace LoopVectorizationUtils
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONUTILS_H