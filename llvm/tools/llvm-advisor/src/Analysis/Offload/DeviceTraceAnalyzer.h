//===--- DeviceTraceAnalyzer.h - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Reads stored device trace JSON files and extracts kernel execution events.
class DeviceTraceAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "offload.device.trace"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
