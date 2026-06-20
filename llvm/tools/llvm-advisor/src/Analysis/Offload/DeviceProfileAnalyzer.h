//===--- DeviceProfileAnalyzer.h - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Parses rocprof kernel-dispatch CSV output files found in the working
// directory. Aggregates per-kernel timing statistics (mean, min, max, total
// duration) from DispatchNs/BeginNs/EndNs columns — no external tool needed.
class DeviceProfileAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override {
    return "offload.device.profile";
  }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
