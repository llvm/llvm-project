//===--- LoopNesting.h - LLVM Advisor ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class LoopNestingInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::LoopNesting; }
  StringRef getName() const override { return "loop_nesting"; }
  StringRef getDescription() const override {
    return "Identifies deeply nested loops and functions with high loop counts";
  }
  StringRef getRequiredCapability() const override { return "llvm.loop_info"; }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
