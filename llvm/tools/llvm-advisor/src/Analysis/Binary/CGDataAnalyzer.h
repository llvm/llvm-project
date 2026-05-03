//===--- CGDataAnalyzer.h - LLVM Advisor ---------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#pragma once
#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Reads CodeGenData embedded in an object file or a .cgdata file in-process
// using the LLVM CGData reader API. Extracts outlined hash trees and stable
// function similarity maps.
class CGDataAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.cgdata"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
