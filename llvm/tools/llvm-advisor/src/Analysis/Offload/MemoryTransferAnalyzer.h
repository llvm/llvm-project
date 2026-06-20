//===--- MemoryTransferAnalyzer.h - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Parses rocprof JSON trace or HIP memory-copy CSV output to extract memory
// transfer events. Reports direction, size, bandwidth, and aggregated totals
// without invoking rocprof — operates on pre-existing output files.
class MemoryTransferAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override {
    return "offload.memory.transfer";
  }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
