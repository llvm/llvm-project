//===- llvm/CodeGen/IRTracker.h - IR tracker recorder -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR tracker instrumentation. Records per-pass IR/MIR snapshots and
// instruction-level changes into a TSV stream when ``-ir-tracker-output`` is
// set. External tooling under ``llvm/tools/ir-tracker/`` consumes the
// recorded stream to query how IR evolves through the optimization pipeline.
//
// Two hooks feed the same underlying recorder so a single compilation can
// capture both the new-pass-manager IR pipeline (via PassInstrumentation
// callbacks) and the legacy-pass-manager machine pipeline (via a
// MachineFunctionPass inserted by TargetPassConfig::addMachinePostPasses).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_IRTRACKER_H
#define LLVM_CODEGEN_IRTRACKER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineFunctionPass;
class PassInstrumentationCallbacks;

/// Register the IR tracker's new-PM PassInstrumentation callbacks. No-op
/// when ``-ir-tracker-output`` is not set.
LLVM_ABI void registerIRTrackerCallbacks(PassInstrumentationCallbacks &PIC);

/// Build a MachineFunctionPass that records the post-pass MIR snapshot for
/// the legacy pass manager. ``Banner`` is expected in the
/// ``"After <pass>"`` form; the leading ``"After "`` is stripped. Returns
/// nullptr when ``-ir-tracker-output`` is not set.
LLVM_ABI MachineFunctionPass *createIRTrackerMIRPass(StringRef Banner);

} // namespace llvm

#endif // LLVM_CODEGEN_IRTRACKER_H
