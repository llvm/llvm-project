//===--- DebugInfo.h - LLVM Advisor --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class DebugInfoInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::DebugInfo; }
  StringRef getName() const override { return "debug_info"; }
  StringRef getDescription() const override {
    return "DWARF coverage assessment: compile units, DWO version, debug "
           "presence";
  }
  StringRef getRequiredCapability() const override {
    return "llvm.debug.summary";
  }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
