//===--- HeaderDepth.h - LLVM Advisor ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class HeaderDepthInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::HeaderDepth; }
  StringRef getName() const override { return "header_depth"; }
  StringRef getDescription() const override {
    return "Include depth analysis: deepest chains and most-included headers";
  }
  StringRef getRequiredCapability() const override {
    return "build.dependency.headers";
  }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
