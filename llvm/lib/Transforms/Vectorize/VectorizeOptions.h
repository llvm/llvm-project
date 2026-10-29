//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_VECTORIZEOPTIONS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_VECTORIZEOPTIONS_H

#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include <optional>

namespace llvm {
enum class LoopIdiomVectorizeStyle;
enum class TailFoldingStyle;

/// Option tail-folding-policy controls the tail-folding strategy and lists all
/// available options. The vectorizer will attempt to fold the tail-loop into
/// the vector loop (main/epilogue loops) and predicate the instructions
/// accordingly. If tail-folding fails, there are different fallback strategies
/// depending on these values:
enum class TailFoldingPolicyTy { None = 0, PreferFoldTail, MustFoldTail };

// The -sbvec-stop-at and -sbvec-stop-bndl value that disables the limit.
constexpr unsigned long SBVecStopDisabled = ~0UL;

#ifdef EXPENSIVE_CHECKS
constexpr bool VPlanVerifyEachDefault = true;
#else
constexpr bool VPlanVerifyEachDefault = false;
#endif
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "VectorizeOptions.inc"

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_VECTORIZEOPTIONS_H
