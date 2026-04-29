//===------------------- CapabilitySpec.cpp - LLVM Advisor
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

#include "Capability/CapabilitySpec.h"

using namespace llvm;
using namespace llvm::advisor;

static json::Array stringsToJSON(ArrayRef<std::string> Values) {
  json::Array Out;
  for (const std::string &Value : Values)
    Out.push_back(Value);
  return Out;
}

json::Value llvm::advisor::toJSON(const CapabilitySpec &Spec) {
  return json::Object{{"id", Spec.ID},
                      {"capability_id", Spec.ID},
                      {"name", Spec.Name},
                      {"description", Spec.Description},
                      {"version", Spec.Version},
                      {"runner", Spec.Runner},
                      {"summary", Spec.Summary},
                      {"execution_mode", Spec.ExecutionMode},
                      {"cost_class", Spec.CostClass},
                      {"readiness", Spec.Readiness},
                      {"readiness_level", Spec.Readiness},
                      {"dependencies", stringsToJSON(Spec.Dependencies)},
                      {"depends_on", stringsToJSON(Spec.Dependencies)},
                      {"required_inputs", stringsToJSON(Spec.RequiredInputs)},
                      {"produces", stringsToJSON(Spec.Produces)},
                      {"supports_scope", stringsToJSON(Spec.SupportsScope)},
                      {"allowed_tools", stringsToJSON(Spec.AllowedTools)}};
}
