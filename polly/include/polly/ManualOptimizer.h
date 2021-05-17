//===------ ManualOptimizer.h ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handle pragma/metadata-directed transformations.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_MANUALOPTIMIZER_H
#define POLLY_MANUALOPTIMIZER_H

#include "isl/isl-noexceptions.h"

namespace polly {
class Scop;

/// Apply loop-transformation metadata.
///
/// The loop metadata are taken from mark-nodes in @sched. These nodes have been
/// added by ScopBuilder when creating a schedule for a loop with an attach
/// LoopID.
///
/// @param S     The SCoP for @p Sched.
/// @param Sched The input schedule to apply the directives on.
///
/// @return The transformed schedule with all mark-nodes with loop
///         transformations applied. Returns NULL in case of an error or @p
///         Sched itself if no transformation has been applied.
isl::schedule applyManualTransformations(Scop *S, isl::schedule Sched);
} // namespace polly

#endif /* POLLY_MANUALOPTIMIZER_H */
