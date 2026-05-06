//===--- SectionSizes.h - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class SectionSizesInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::SectionSizes; }
  StringRef getName() const override { return "section_sizes"; }
  StringRef getDescription() const override {
    return "Binary section breakdown: text vs data vs debug size distribution";
  }
  StringRef getRequiredCapability() const override {
    return "llvm.obj.readobj";
  }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
