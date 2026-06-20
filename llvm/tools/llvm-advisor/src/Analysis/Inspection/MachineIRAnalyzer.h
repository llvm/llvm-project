//===------------------- MachineIRAnalyzer.h - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// Extracts structural metrics from LLVM IR that are relevant to Machine IR
// lowering (phi nodes, memory operations, instruction counts, etc.).
// Note: this operates on LLVM IR, not Machine IR.
class MachineIRAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.machine_ir"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

} // namespace llvm::advisor
