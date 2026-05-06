//===--- CompilationFlow.h - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class CompilationFlowInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::CompilationFlow; }
  StringRef getName() const override { return "compilation_flow"; }
  StringRef getDescription() const override {
    return "Breaks down compilation time by phase (frontend/optimizer/codegen)";
  }
  StringRef getRequiredCapability() const override {
    return "build.time_trace";
  }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
