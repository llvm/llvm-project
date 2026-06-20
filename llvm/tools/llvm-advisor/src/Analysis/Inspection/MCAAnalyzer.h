//===--- MCAAnalyzer.h - LLVM Advisor ------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Stub: in-process LLVM MCA simulation is not implemented for this LLVM API
// version.  The analyzer reports unavailable and directs users to run
// llvm-mca externally on the assembly artifact.
class MCAAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.mca.report"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
