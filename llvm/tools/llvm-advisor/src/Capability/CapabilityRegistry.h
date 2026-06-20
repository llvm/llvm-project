//===------------------- CapabilityRegistry.h - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilityRegistry in Capability
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Analysis/AnalyzerBase.h"
#include "Capability/CapabilitySpec.h"

namespace llvm::advisor {

class CapabilityRegistry {
public:
  Error loadDirectory(StringRef ConfigDir);
  Error loadFile(StringRef ConfigFile);
  Error addSpec(CapabilitySpec Spec);
  Expected<CapabilitySpec> getSpec(StringRef ID) const;
  SmallVector<CapabilitySpec, 32> listSpecs() const;

  Error addRunner(std::unique_ptr<CapabilityRunner> Runner);
  Error addRunner(StringRef RunnerKind,
                  std::unique_ptr<CapabilityRunner> Runner);
  CapabilityRunner *getRunner(StringRef ID) const;
  CapabilityRunner *getRunner(const CapabilitySpec &Spec) const;
  std::unique_ptr<CapabilityRunner>
  createDeclarativeRunner(const CapabilitySpec &Spec) const;
  void addBuiltinRunners();

private:
  StringMap<CapabilitySpec> Specs;
  StringMap<std::unique_ptr<CapabilityRunner>> Runners;
  StringMap<CapabilityRunner *> RunnerKinds;
};

} // namespace llvm::advisor
