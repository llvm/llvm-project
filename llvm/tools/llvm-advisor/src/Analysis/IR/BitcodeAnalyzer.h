//===------------------- BitcodeAnalyzer.h - LLVM Advisor ===================//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

class BitcodeAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.bcanalyzer"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor