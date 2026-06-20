//===--- LinkerMapAnalyzer.h - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Parses linker map files (LLD text format or GNU ld format) produced during
// the link step. Reports section sizes, top symbols by size, and a binary
// layout summary — all derived in-process without invoking any external tool.
class LinkerMapAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "lld.mapfile"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
