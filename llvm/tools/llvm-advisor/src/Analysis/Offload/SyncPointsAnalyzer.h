//===--- SyncPointsAnalyzer.h - LLVM Advisor -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Parses rocprof JSON trace files to identify HIP/HSA synchronization events
// (hipDeviceSynchronize, hipStreamSynchronize, hipEventSynchronize, barriers).
// Reports sync-point counts, total wait time, and the top stalling calls —
// all derived in-process from the trace file without re-running rocprof.
class SyncPointsAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "offload.sync.points"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
