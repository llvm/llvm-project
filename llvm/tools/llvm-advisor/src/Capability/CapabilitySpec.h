//===------------------- CapabilitySpec.h - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilitySpec in Capability
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

struct CapabilitySpec {
  std::string ID;
  std::string Name;
  std::string Description;
  std::string Version = "1";
  std::string Runner = "generic.unavailable";
  std::string Summary;
  std::string ExecutionMode = "library";
  std::string CostClass = "cheap";
  SmallVector<std::string, 4> Dependencies;
  SmallVector<std::string, 4> RequiredInputs;
  SmallVector<std::string, 4> Produces;
  SmallVector<std::string, 4> SupportsScope;
  SmallVector<std::string, 4> AllowedTools;
  std::string Readiness = "L0";
};

struct CapabilityNode {
  CapabilitySpec Spec;
  SmallVector<std::string, 4> Inputs;
};

json::Value toJSON(const CapabilitySpec &Spec);

} // namespace llvm::advisor
