//===-- Coroutines.h - Coroutine Transformations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declare accessor functions for coroutine lowering passes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COROUTINES_H
#define LLVM_TRANSFORMS_COROUTINES_H

namespace llvm {

class Pass;
class PassManagerBuilder;

/// Add all coroutine passes to appropriate extension points.
void addCoroutinePassesToExtensionPoints(PassManagerBuilder &Builder);

/// Lower coroutine intrinsics that are not needed by later passes.
Pass *createCoroEarlyLegacyPass();

/// Split up coroutines into multiple functions driving their state machines.
Pass *createCoroSplitLegacyPass(bool ReuseFrameSlot = false);

/// Analyze coroutines use sites, devirtualize resume/destroy calls and elide
/// heap allocation for coroutine frame where possible.
Pass *createCoroElideLegacyPass();

/// Lower all remaining coroutine intrinsics.
Pass *createCoroCleanupLegacyPass();

}

#endif
