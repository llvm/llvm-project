//===------------------- CapabilityExecutor.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilityExecutor in Capability
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Analysis/AnalyzerBase.h"
#include "Capability/CapabilityCache.h"
#include "Capability/CapabilityRegistry.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {

class CapabilityExecutor {
public:
  CapabilityExecutor(const CapabilityRegistry &Registry,
                     StorageManager &Storage)
      : Registry(Registry), Storage(Storage), Cache(Storage.results()) {}

  Expected<json::Array> execute(ArrayRef<CapabilityNode> Nodes,
                                const CapabilityContext &Context);

private:
  Error materializeEntities(const CapabilitySpec &Spec,
                            const CapabilityContext &Context, StringRef RunKey,
                            StringRef ResultID, const json::Value &Value);

  const CapabilityRegistry &Registry;
  StorageManager &Storage;
  CapabilityCache Cache;
};

} // namespace llvm::advisor
