//===--- OffloadBinaryAnalyzer.h - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Inspects offload sections embedded in an object file using LLVM's
// OffloadBinary API. Reports image kind, target triple, arch, and image size
// for each embedded offload entry.
class OffloadBinaryAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override {
    return "offload.binary.inspect";
  }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
