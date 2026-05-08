//===--- LTOAnalyzer.h - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Reads the LTO / ThinLTO summary index from a bitcode file in-process using
// LLVM's BitcodeReader and ModuleSummaryIndex APIs. Reports module count,
// function/global counts, and cross-module import/export statistics.
class LTOAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.lto.summary"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
