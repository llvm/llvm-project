//===------------------- CapabilityExecutor.cpp - LLVM Advisor -------------------===//
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

#include "Capability/CapabilityExecutor.h"
#include "Utils/Hashing.h"
#include "Utils/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

static void initCommonEntityFields(EntityRecord &Entity,
                                   const CapabilityContext &Context) {
  Entity.SnapshotID = Context.Unit.SnapshotID;
  Entity.UnitID = Context.Unit.ID;
  Entity.OwnerID = Context.Unit.ID;
}

Error CapabilityExecutor::materializeEntities(const CapabilitySpec &Spec,
                                              const CapabilityContext &Context,
                                              StringRef RunKey,
                                              StringRef ResultID,
                                              const json::Value &Value) {
  const json::Object *Object = Value.getAsObject();
  if (!Object)
    return Error::success();

  SmallVector<std::string, 4> Kinds = Spec.Produces;
  if (Kinds.empty()) {
    if (std::optional<StringRef> Kind = Object->getString("kind"))
      Kinds.push_back(Kind->str());
    else
      Kinds.push_back(Spec.ID);
  }

  for (StringRef Kind : Kinds) {
    EntityRecord Representation;
    Representation.Kind = "representation";
    Representation.ID = "repr_" + hashString((RunKey + Kind).str());
    initCommonEntityFields(Representation, Context);
    Representation.Data = json::Object{{"kind", Kind.str()},
                                       {"representation_id", Representation.ID},
                                       {"owner_id", Context.Unit.ID},
                                       {"owner_type", "unit"},
                                       {"capability_id", Spec.ID},
                                       {"capability_version", Spec.Version},
                                       {"content_type", "application/json"},
                                       {"content_address", ResultID.str()},
                                       {"readiness_level", Spec.Readiness},
                                       {"cost_class", Spec.CostClass},
                                       {"materialization_policy", "eager"}};
    if (Error Err = Storage.metadata().putEntity(Representation))
      return Err;
  }

  if (const json::Array *Findings = Object->getArray("findings")) {
    for (const json::Value &FindingValue : *Findings) {
      const json::Object *FindingObject = FindingValue.getAsObject();
      if (!FindingObject)
        continue;
      EntityRecord Finding;
      Finding.Kind = "finding";
      Finding.ID =
          "finding_" + hashString((RunKey + stringifyJSON(FindingValue)).str());
      initCommonEntityFields(Finding, Context);
      Finding.Data = *FindingObject;
      Finding.Data["finding_id"] = Finding.ID;
      Finding.Data["owner_id"] = Context.Unit.ID;
      Finding.Data["owner_type"] = "unit";
      Finding.Data["provenance"] =
          json::Object{{"snapshot_id", Context.Unit.SnapshotID},
                       {"unit_id", Context.Unit.ID},
                       {"capability_id", Spec.ID},
                       {"capability_version", Spec.Version},
                       {"toolchain_version", Context.Unit.ToolchainVersion}};
      if (Error Err = Storage.metadata().putEntity(Finding))
        return Err;
    }
  }

  if (const json::Array *Mappings = Object->getArray("mappings")) {
    for (const json::Value &MappingValue : *Mappings) {
      const json::Object *MappingObject = MappingValue.getAsObject();
      if (!MappingObject)
        continue;
      EntityRecord Mapping;
      Mapping.Kind = "mapping";
      Mapping.ID =
          "map_" + hashString((RunKey + stringifyJSON(MappingValue)).str());
      initCommonEntityFields(Mapping, Context);
      Mapping.Data = *MappingObject;
      Mapping.Data["mapping_id"] = Mapping.ID;
      if (Error Err = Storage.metadata().putEntity(Mapping))
        return Err;
    }
  }

  return Error::success();
}

Expected<json::Array>
CapabilityExecutor::execute(ArrayRef<CapabilityNode> Nodes,
                            const CapabilityContext &Context) {
  json::Array Results;
  std::string InputDigest = hashString(Context.Unit.SourceContentHash + "\0" +
                                       Context.Unit.CommandFingerprint);
  for (const CapabilityNode &Node : Nodes) {
    std::string RunKey = computeCapabilityRunKey(
        Context.Unit, Node.Spec.ID, Node.Spec.Version, InputDigest);
    if (Cache.contains(RunKey)) {
      Expected<std::string> ID = Cache.get(RunKey);
      if (!ID)
        return ID.takeError();
      Expected<std::string> Blob = Storage.blobs().get(*ID);
      if (!Blob)
        return Blob.takeError();
      Expected<json::Value> Cached = json::parse(*Blob);
      if (!Cached)
        return Cached.takeError();
      Results.push_back(json::Object{{"capability", Node.Spec.ID},
                                     {"run_key", RunKey},
                                     {"result_id", *ID},
                                     {"cache", "hit"},
                                     {"value", std::move(*Cached)}});
      continue;
    }

    CapabilityRunner *Runner = Registry.getRunner(Node.Spec);
    std::unique_ptr<CapabilityRunner> DeclarativeRunner;
    if (!Runner) {
      DeclarativeRunner = Registry.createDeclarativeRunner(Node.Spec);
      Runner = DeclarativeRunner.get();
    }
    if (!Runner)
      return createStringError(inconvertibleErrorCode(),
                               "no runner '%s' for capability %s",
                               Node.Spec.Runner.c_str(), Node.Spec.ID.c_str());

    Expected<std::unique_ptr<CapabilityResult>> Result = Runner->run(Context);
    if (!Result) {
      Results.push_back(json::Object{
          {"capability", Node.Spec.ID},
          {"run_key", RunKey},
          {"cache", "error"},
          {"value", json::Object{{"capability", Node.Spec.ID},
                                 {"unit_id", Context.Unit.ID},
                                 {"available", false},
                                 {"reason", toString(Result.takeError())}}}});
      continue;
    }
    json::Value Value = (*Result)->toJSON();
    std::string ResultID;
    std::string CacheStatus = "miss";
    if (Expected<std::string> ID = Cache.put(RunKey, Value)) {
      ResultID = *ID;
      if (Error Err =
              materializeEntities(Node.Spec, Context, RunKey, ResultID, Value))
        return std::move(Err);
    } else {
      // Cache write failures are non-fatal: the capability ran successfully,
      // we simply cannot persist the result for future reuse.
      consumeError(ID.takeError());
      CacheStatus = "miss (uncached)";
    }
    Results.push_back(json::Object{{"capability", Node.Spec.ID},
                                   {"run_key", RunKey},
                                   {"result_id", ResultID},
                                   {"cache", CacheStatus},
                                   {"value", std::move(Value)}});
  }
  return Results;
}
