//===------------------- CapabilityCache.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilityCache in Capability
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/ResultStore.h"

namespace llvm::advisor {

class CapabilityCache {
public:
  explicit CapabilityCache(ResultStore &Results) : Results(Results) {}

  bool contains(StringRef RunKey) const { return Results.contains(RunKey); }
  Expected<std::string> get(StringRef RunKey) const {
    return Results.get(RunKey);
  }
  Expected<std::string> put(StringRef RunKey, const json::Value &Value) {
    return Results.put(RunKey, Value);
  }

private:
  ResultStore &Results;
};

} // namespace llvm::advisor
