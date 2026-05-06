//===--- DiagnosticDelta.h - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class DiagnosticDeltaInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::DiagnosticDelta; }
  StringRef getName() const override { return "diagnostic_delta"; }
  StringRef getDescription() const override {
    return "Compares compiler diagnostics against a baseline snapshot";
  }
  StringRef getRequiredCapability() const override {
    return "clang.diag.summary";
  }
  bool requiresBaseline() const override { return true; }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
