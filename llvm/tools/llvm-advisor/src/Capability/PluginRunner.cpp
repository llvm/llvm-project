//===------------------- PluginRunner.cpp - LLVM Advisor -------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Part of PluginRunner in Capability
//
//===----------------------------------------------------------------------===//

#include "Capability/PluginRunner.h"
#include "Utils/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

PluginRunner::PluginRunner(StringRef CapID,
                           decltype(&llvm_advisor_run_capability) RunFn,
                           decltype(&llvm_advisor_free_result) FreeFn)
    : CapID(CapID.str()), RunFn(RunFn), FreeFn(FreeFn) {}

Expected<std::unique_ptr<CapabilityResult>>
PluginRunner::run(const CapabilityContext &Context) {
  AdvisorRunContext Ctx{};
  Ctx.unit_id = Context.Unit.ID.c_str();
  Ctx.snapshot_id = Context.Unit.SnapshotID.c_str();
  Ctx.data_dir = Context.WorkingDirectory.c_str();
  // Plugins are expected to be self-contained; blob callbacks are optional.
  Ctx.read_blob = nullptr;
  Ctx.free_blob = nullptr;
  Ctx.cancellation_token = nullptr;

  AdvisorCapabilityResult Result = RunFn(CapID.c_str(), &Ctx);
  std::string ErrMsg = Result.error_message ? Result.error_message : "";
  if (!Result.success) {
    if (FreeFn)
      FreeFn(&Result);
    return createStringError(inconvertibleErrorCode(), "plugin error: %s",
                             ErrMsg.c_str());
  }

  json::Value Value = json::Object{
      {"capability", CapID},
      {"unit_id", Context.Unit.ID},
      {"available", true},
  };
  if (Result.output_json) {
    Expected<json::Value> Parsed = json::parse(Result.output_json);
    if (!Parsed) {
      if (FreeFn)
        FreeFn(&Result);
      return Parsed.takeError();
    }
    if (const json::Object *Obj = Parsed->getAsObject()) {
      // Merge plugin output into the result object.
      json::Object Out = *Obj;
      Out["capability"] = CapID;
      Out["unit_id"] = Context.Unit.ID;
      Out["available"] = true;
      Value = std::move(Out);
    }
  }
  if (FreeFn)
    FreeFn(&Result);
  return std::make_unique<JSONCapabilityResult>(std::move(Value));
}
