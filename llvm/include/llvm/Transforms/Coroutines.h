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

// CoroEarly pass marks every function that has coro.begin with a string
// attribute "coroutine.presplit"="0". CoroSplit pass processes the coroutine
// twice. First, it lets it go through complete IPO optimization pipeline as a
// single function. It forces restart of the pipeline by inserting an indirect
// call to an empty function "coro.devirt.trigger" which is devirtualized by
// CoroElide pass that triggers a restart of the pipeline by CGPassManager.
// When CoroSplit pass sees the same coroutine the second time, it splits it up,
// adds coroutine subfunctions to the SCC to be processed by IPO pipeline.
#define CORO_PRESPLIT_ATTR "coroutine.presplit"
#define UNPREPARED_FOR_SPLIT "0"
#define PREPARED_FOR_SPLIT "1"

class Pass;
class PassManagerBuilder;

/// Add all coroutine passes to appropriate extension points.
void addCoroutinePassesToExtensionPoints(PassManagerBuilder &Builder);

/// Lower coroutine intrinsics that are not needed by later passes.
Pass *createCoroEarlyLegacyPass();

/// Split up coroutines into multiple functions driving their state machines.
Pass *createCoroSplitLegacyPass();

/// Analyze coroutines use sites, devirtualize resume/destroy calls and elide
/// heap allocation for coroutine frame where possible.
Pass *createCoroElideLegacyPass();

/// Lower all remaining coroutine intrinsics.
Pass *createCoroCleanupLegacyPass();

}

#endif
