//===--- StaticAnalysisAnalyzer.h - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clang static analyzer frontend action wrapper.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

class StaticAnalysisAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "clang.static_analysis"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
