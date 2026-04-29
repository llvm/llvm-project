//===- llvm/Passes/IRTrackerInstrumentation.h - IR tracker ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR tracker instrumentation. Records per-pass IR snapshots and instruction-
// level changes into a TSV stream when ``-ir-tracker-output`` is set.
// External tooling under ``llvm/tools/ir-tracker/`` consumes the recorded
// stream to query how IR evolves through the optimization pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_IRTRACKERINSTRUMENTATION_H
#define LLVM_PASSES_IRTRACKERINSTRUMENTATION_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class PassInstrumentationCallbacks;

/// Wires the IR tracker into the new pass manager via ``PassInstrumentation``
/// callbacks. ``registerCallbacks`` is a no-op when the recorder's CLI flag
/// is not set, so embedding this in ``StandardInstrumentations`` is free for
/// users who do not opt in.
class IRTrackerInstrumentation {
public:
  LLVM_ABI void registerCallbacks(PassInstrumentationCallbacks &PIC);
};

} // namespace llvm

#endif // LLVM_PASSES_IRTRACKERINSTRUMENTATION_H
