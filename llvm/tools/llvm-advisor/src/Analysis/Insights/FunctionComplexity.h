//===--- FunctionComplexity.h - LLVM Advisor -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class FunctionComplexityInsight final : public Insight {
public:
  InsightKind getKind() const override {
    return InsightKind::FunctionComplexity;
  }
  StringRef getName() const override { return "function_complexity"; }
  StringRef getDescription() const override {
    return "Ranks functions by instruction count and identifies hotspots";
  }
  StringRef getRequiredCapability() const override {
    return "llvm.ir.function_stats";
  }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
