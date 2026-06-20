//===------------------- PluginRunner.h - LLVM Advisor -------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// CapabilityRunner that delegates to an out-of-tree plugin via the stable C
// ABI defined in llvm-advisor-plugin.h.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Analysis/AnalyzerBase.h"
#include "Capability/llvm-advisor-plugin.h"

namespace llvm::advisor {

/// Wraps a plugin's llvm_advisor_run_capability function so it can be
/// registered as a first-class runner in CapabilityRegistry.
class PluginRunner final : public CapabilityRunner {
public:
  PluginRunner(StringRef CapID, decltype(&llvm_advisor_run_capability) RunFn,
               decltype(&llvm_advisor_free_result) FreeFn);

  StringRef getCapabilityID() const override { return CapID; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;

private:
  std::string CapID;
  decltype(&llvm_advisor_run_capability) RunFn;
  decltype(&llvm_advisor_free_result) FreeFn;
};

} // namespace llvm::advisor
