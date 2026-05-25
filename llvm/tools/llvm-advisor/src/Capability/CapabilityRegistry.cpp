//===--- CapabilityRegistry.cpp - LLVM Advisor ---------------------------===//
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

#include "Capability/CapabilityRegistry.h"
#include "Analysis/IR/RemarksAnalyzer.h"
#include "Analysis/IR/RemarksMixAnalyzer.h"
#include "Analysis/IR/RemarksRelationalAnalyzer.h"
#include "Analysis/IR/RemarksSizeDiffAnalyzer.h"
#include "Analysis/Inspection/RemarksDetailAnalyzer.h"
#include "Utils/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static SmallVector<std::string, 4>
getStringOrObjectFieldArray(const json::Object &Object, StringRef Key,
                            StringRef Field) {
  SmallVector<std::string, 4> Out;
  const json::Array *Array = Object.getArray(Key);
  if (!Array)
    return Out;
  for (const json::Value &Value : *Array) {
    if (std::optional<StringRef> String = Value.getAsString()) {
      Out.push_back(String->str());
      continue;
    }
    const json::Object *Item = Value.getAsObject();
    if (!Item)
      continue;
    if (std::optional<StringRef> String = Item->getString(Field))
      Out.push_back(String->str());
  }
  return Out;
}

static CapabilitySpec specFromJSON(const json::Object &Object) {
  CapabilitySpec Spec;
  if (std::optional<StringRef> ID = Object.getString("id"))
    Spec.ID = ID->str();
  if (std::optional<StringRef> ID = Object.getString("capability_id"))
    Spec.ID = ID->str();
  if (std::optional<StringRef> Name = Object.getString("name"))
    Spec.Name = Name->str();
  if (std::optional<StringRef> Description = Object.getString("description"))
    Spec.Description = Description->str();
  if (std::optional<StringRef> Version = Object.getString("version"))
    Spec.Version = Version->str();
  if (std::optional<StringRef> Runner = Object.getString("runner"))
    Spec.Runner = Runner->str();
  if (std::optional<StringRef> Summary = Object.getString("summary"))
    Spec.Summary = Summary->str();
  if (std::optional<StringRef> Mode = Object.getString("execution_mode"))
    Spec.ExecutionMode = Mode->str();
  if (std::optional<StringRef> Cost = Object.getString("cost_class"))
    Spec.CostClass = Cost->str();
  if (std::optional<StringRef> Readiness = Object.getString("readiness"))
    Spec.Readiness = Readiness->str();
  if (std::optional<StringRef> Readiness = Object.getString("readiness_level"))
    Spec.Readiness = Readiness->str();
  Spec.Dependencies = getStringArray(Object, "dependencies");
  if (Spec.Dependencies.empty())
    Spec.Dependencies = getStringArray(Object, "depends_on");
  Spec.RequiredInputs = getStringArray(Object, "required_inputs");
  Spec.Produces = getStringOrObjectFieldArray(Object, "produces", "kind");
  Spec.SupportsScope = getStringArray(Object, "supports_scope");
  Spec.AllowedTools =
      getStringOrObjectFieldArray(Object, "allowed_tools", "name");
  return Spec;
}

Error CapabilityRegistry::loadDirectory(StringRef ConfigDir) {
  std::error_code EC;
  for (sys::fs::directory_iterator I(ConfigDir, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (!sys::path::extension(I->path()).equals_insensitive(".json"))
      continue;
    if (Error Err = loadFile(I->path()))
      return Err;
  }
  if (EC)
    return createStringError(EC, Twine("cannot read capability directory '") +
                                      ConfigDir + "'");
  return Error::success();
}

Error CapabilityRegistry::loadFile(StringRef ConfigFile) {
  Expected<json::Value> Value = parseJSONFile(ConfigFile);
  if (!Value)
    return Value.takeError();

  if (const json::Array *Array = Value->getAsArray()) {
    for (const json::Value &Item : *Array) {
      const json::Object *Object = Item.getAsObject();
      if (!Object)
        return createStringError(inconvertibleErrorCode(),
                                 "capability array item is not an object");
      CapabilitySpec Spec = specFromJSON(*Object);
      if (Spec.ID.empty())
        return createStringError(inconvertibleErrorCode(),
                                 "capability spec missing id");
      if (Error Err = addSpec(std::move(Spec)))
        return Err;
    }
    return Error::success();
  }

  const json::Object *Object = Value->getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(),
                             "capability spec is not an object or array");

  if (const json::Array *Capabilities = Object->getArray("capabilities")) {
    for (const json::Value &Item : *Capabilities) {
      const json::Object *SpecObject = Item.getAsObject();
      if (!SpecObject)
        return createStringError(inconvertibleErrorCode(),
                                 "capability array item is not an object");
      CapabilitySpec Spec = specFromJSON(*SpecObject);
      if (Spec.ID.empty())
        return createStringError(inconvertibleErrorCode(),
                                 "capability spec missing id");
      if (Error Err = addSpec(std::move(Spec)))
        return Err;
    }
    return Error::success();
  }

  CapabilitySpec Spec = specFromJSON(*Object);
  if (Spec.ID.empty())
    return createStringError(inconvertibleErrorCode(),
                             "capability spec missing id");
  return addSpec(std::move(Spec));
}

Error CapabilityRegistry::addSpec(CapabilitySpec Spec) {
  if (Spec.ID.empty())
    return createStringError(inconvertibleErrorCode(), "empty capability id");
  Specs[Spec.ID] = std::move(Spec);
  return Error::success();
}

Expected<CapabilitySpec> CapabilityRegistry::getSpec(StringRef ID) const {
  StringMap<CapabilitySpec>::const_iterator I = Specs.find(ID);
  if (I == Specs.end())
    return createStringError(inconvertibleErrorCode(),
                             Twine("unknown capability: ") + ID);
  return I->second;
}

SmallVector<CapabilitySpec, 32> CapabilityRegistry::listSpecs() const {
  SmallVector<CapabilitySpec, 32> Out;
  for (const StringMapEntry<CapabilitySpec> &Entry : Specs)
    Out.push_back(Entry.second);
  return Out;
}

Error CapabilityRegistry::addRunner(std::unique_ptr<CapabilityRunner> Runner) {
  if (!Runner)
    return createStringError(inconvertibleErrorCode(),
                             "null capability runner");
  StringRef ID = Runner->getCapabilityID();
  if (Runners.contains(ID))
    return createStringError(inconvertibleErrorCode(),
                             Twine("runner already registered: ") + ID);
  Runners[ID] = std::move(Runner);
  RunnerKinds[ID] = Runners[ID].get();
  return Error::success();
}

Error CapabilityRegistry::addRunner(StringRef RunnerKind,
                                    std::unique_ptr<CapabilityRunner> Runner) {
  if (!Runner)
    return createStringError(inconvertibleErrorCode(),
                             "null capability runner");
  std::string ID = Runner->getCapabilityID().str();
  if (Runners.contains(ID))
    return createStringError(inconvertibleErrorCode(),
                             Twine("runner already registered: ") + ID);
  Runners[ID] = std::move(Runner);
  RunnerKinds[RunnerKind] = Runners[ID].get();
  return Error::success();
}

CapabilityRunner *CapabilityRegistry::getRunner(StringRef ID) const {
  StringMap<std::unique_ptr<CapabilityRunner>>::const_iterator I =
      Runners.find(ID);
  if (I == Runners.end())
    return nullptr;
  return I->second.get();
}

CapabilityRunner *
CapabilityRegistry::getRunner(const CapabilitySpec &Spec) const {
  StringMap<CapabilityRunner *>::const_iterator Kind =
      RunnerKinds.find(Spec.Runner);
  if (Kind != RunnerKinds.end())
    return Kind->second;
  return getRunner(Spec.ID);
}

std::unique_ptr<CapabilityRunner>
CapabilityRegistry::createDeclarativeRunner(const CapabilitySpec &Spec) const {
  if (Spec.Runner != "generic.unavailable")
    return nullptr;
  StringRef Summary = Spec.Summary.empty()
                          ? "capability is declared but has no runner"
                          : StringRef(Spec.Summary);
  return std::make_unique<SimpleAnalyzer>(Spec.ID, Summary);
}

void CapabilityRegistry::addBuiltinRunners() {
  consumeError(addRunner("builtin.remarks_summary",
                         std::make_unique<RemarksAnalyzer>()));
  consumeError(addRunner("builtin.remarks_mix",
                         std::make_unique<RemarksMixAnalyzer>()));
  consumeError(addRunner("builtin.remarks_size_diff",
                         std::make_unique<RemarksSizeDiffAnalyzer>()));
  consumeError(addRunner("builtin.remarks_relational",
                         std::make_unique<RemarksRelationalAnalyzer>()));
  consumeError(addRunner("builtin.remarks_detail",
                         std::make_unique<RemarksDetailAnalyzer>()));
}
