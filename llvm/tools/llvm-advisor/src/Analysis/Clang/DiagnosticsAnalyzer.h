//===--- DiagnosticsAnalyzer.h - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clang diagnostics summary analyzer.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

class DiagnosticsAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "clang.diag.summary"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
