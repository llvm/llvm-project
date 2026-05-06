//===------------------- CoreClient.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared core access used by all clients (CLI, HTTP, LSP).
// Provides unified API for snapshots, units, queries, and results.
//
//===----------------------------------------------------------------------===//

#include "Client/CoreClient.h"
#include "Analysis/Insights/InsightBase.h"
#include "Capability/CapabilityExecutor.h"
#include "Capability/CapabilityPlanner.h"
#include "Capability/CapabilityScheduler.h"
#include "Compare/CompareEngine.h"
#include "Core/CaptureCore.h"
#include "Core/SnapshotManager.h"
#include "Runtime/CoverageIngestor.h"
#include "Runtime/MemProfIngestor.h"
#include "Runtime/OffloadRuntime.h"
#include "Runtime/PGOInstrIngestor.h"
#include "Runtime/PGOSampleIngestor.h"
#include "Runtime/RuntimeCorrelator.h"
#include "Runtime/SancovIngestor.h"
#include "Runtime/SanitizerIngestor.h"
#include "Runtime/XRayIngestor.h"
#include "Utils/Hashing.h"
#include "Utils/JSON.h"
#include "Utils/Normalization.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::advisor;

static std::string resolveObjectPath(const UnitRecord &Unit) {
  if (!Unit.ObjectPath.empty() && sys::fs::exists(Unit.ObjectPath))
    return Unit.ObjectPath;
  return resolveOutputPath(Unit.Arguments, Unit.Directory);
}

static std::string resolveIRPath(const UnitRecord &Unit, StringRef ObjectPath) {
  if (!Unit.IRPath.empty() && sys::fs::exists(Unit.IRPath))
    return Unit.IRPath;

  SmallVector<std::string, 8> Candidates;
  auto addCandidates = [&](StringRef Path) {
    if (Path.empty())
      return;
    SmallString<256> LL(Path);
    SmallString<256> BC(Path);
    sys::path::replace_extension(LL, "ll");
    sys::path::replace_extension(BC, "bc");
    Candidates.push_back(LL.str().str());
    Candidates.push_back(BC.str().str());
  };
  addCandidates(ObjectPath);
  addCandidates(Unit.SourcePath);
  for (const std::string &Candidate : Candidates) {
    if (sys::fs::exists(Candidate))
      return Candidate;
  }
  return {};
}

static CapabilityContext makeContext(const UnitRecord &Unit) {
  CapabilityContext Context;
  Context.Unit = Unit;
  Context.SourcePath = Unit.SourcePath;
  Context.WorkingDirectory = Unit.Directory;
  Context.ToolchainVersion = Unit.ToolchainVersion;
  Context.ObjectPath = resolveObjectPath(Unit);
  if (!Context.ObjectPath.empty() && !sys::fs::exists(Context.ObjectPath))
    Context.ObjectPath.clear();
  Context.IRPath = resolveIRPath(Unit, Context.ObjectPath);
  Context.RemarksPath = Unit.RemarksPath;
  return Context;
}

static const json::Object *lookupCapabilityValue(const json::Array &Results,
                                                 StringRef CapabilityID) {
  for (const json::Value &V : Results) {
    const json::Object *O = V.getAsObject();
    if (!O)
      continue;
    std::optional<StringRef> Cap = O->getString("capability");
    if (!Cap || *Cap != CapabilityID)
      continue;
    return O->getObject("value");
  }
  return nullptr;
}

static std::string renderFunctionIRSlice(StringRef IR, StringRef FunctionName) {
  if (FunctionName.empty())
    return IR.str();

  SmallVector<StringRef, 256> Lines;
  IR.split(Lines, '\n');
  std::string Out;
  raw_string_ostream OS(Out);
  bool InFunction = false;
  int Depth = 0;
  for (StringRef Line : Lines) {
    if (!InFunction) {
      if (!Line.contains("define") || !Line.contains(FunctionName))
        continue;
      InFunction = true;
    }
    OS << Line << '\n';
    Depth += Line.count('{');
    Depth -= Line.count('}');
    if (InFunction && Depth <= 0 && Line.trim() == "}")
      break;
  }
  OS.flush();
  return Out;
}

static bool matchesInspectionFilter(StringRef Value, StringRef Filter) {
  return Filter.empty() || Value.contains(Filter);
}

static json::Array filterNamedItems(const json::Array &Items,
                                    const InspectionFilter &Filter) {
  json::Array Filtered;
  int64_t Index = 0;
  for (const json::Value &ItemValue : Items) {
    const json::Object *Item = ItemValue.getAsObject();
    if (!Item)
      continue;
    StringRef Name = Item->getString("name").value_or(
        Item->getString("function").value_or(""));
    if (!matchesInspectionFilter(Name, Filter.Function))
      continue;
    if (!matchesInspectionFilter(Item->getString("pass").value_or(""),
                                 Filter.Pass) &&
        !matchesInspectionFilter(Item->getString("arg").value_or(""),
                                 Filter.Pass))
      continue;
    if (Filter.Index >= 0 && Index++ != Filter.Index)
      continue;
    json::Object Copy = *Item;
    Filtered.push_back(json::Value(std::move(Copy)));
    if (Filter.Index >= 0)
      break;
  }
  return Filtered;
}

static json::Object filterInspectionValue(const json::Object &Value,
                                          const InspectionFilter &Filter) {
  json::Object Out = Value;

  if (const json::Array *Remarks = Value.getArray("remarks")) {
    json::Array Filtered;
    int64_t Index = 0;
    for (const json::Value &Item : *Remarks) {
      const json::Object *Remark = Item.getAsObject();
      if (!Remark)
        continue;
      if (!matchesInspectionFilter(Remark->getString("function").value_or(""),
                                   Filter.Function))
        continue;
      if (!matchesInspectionFilter(Remark->getString("pass").value_or(""),
                                   Filter.Pass))
        continue;
      if (!matchesInspectionFilter(Remark->getString("type").value_or(""),
                                   Filter.Severity))
        continue;
      if (Filter.File != "" &&
          !matchesInspectionFilter(Remark->getObject("location")
                                       ? Remark->getObject("location")
                                             ->getString("file")
                                             .value_or("")
                                       : StringRef(),
                                   Filter.File))
        continue;
      if (Filter.Line >= 0) {
        const json::Object *Loc = Remark->getObject("location");
        if (!Loc || Loc->getInteger("line").value_or(-1) != Filter.Line)
          continue;
      }
      if (Filter.Index >= 0 && Index++ != Filter.Index)
        continue;
      json::Object Copy = *Remark;
      Filtered.push_back(json::Value(std::move(Copy)));
      if (Filter.Index >= 0)
        break;
    }
    Out["remarks"] = std::move(Filtered);
    Out["count"] = static_cast<int64_t>(
        Out.getArray("remarks") ? Out.getArray("remarks")->size() : 0);
  }

  if (const json::Array *Diagnostics = Value.getArray("diagnostics")) {
    json::Array Filtered;
    int64_t Index = 0;
    for (const json::Value &Item : *Diagnostics) {
      const json::Object *Diag = Item.getAsObject();
      if (!Diag)
        continue;
      StringRef Severity = Diag->getString("severity")
                               .value_or(Diag->getString("level").value_or(""));
      if (!matchesInspectionFilter(Severity, Filter.Severity))
        continue;
      if (!matchesInspectionFilter(Diag->getString("file").value_or(""),
                                   Filter.File))
        continue;
      if (Filter.Line >= 0 &&
          Diag->getInteger("line").value_or(-1) != Filter.Line)
        continue;
      if (Filter.Index >= 0 && Index++ != Filter.Index)
        continue;
      json::Object Copy = *Diag;
      Filtered.push_back(json::Value(std::move(Copy)));
      if (Filter.Index >= 0)
        break;
    }
    Out["diagnostics"] = std::move(Filtered);
  }

  if (const json::Array *Functions = Value.getArray("functions"))
    Out["functions"] = filterNamedItems(*Functions, Filter);

  if (const json::Array *Passes = Value.getArray("passes"))
    Out["passes"] = filterNamedItems(*Passes, Filter);

  if (std::optional<StringRef> IR = Value.getString("ir"))
    Out["ir"] = renderFunctionIRSlice(*IR, Filter.Function);

  return Out;
}

static void collectNumericMetrics(const json::Object &Object,
                                  StringMap<double> &Metrics) {
  for (const auto &KV : Object) {
    if (auto Number = KV.second.getAsNumber())
      Metrics[KV.first.str()] = *Number;
    else if (auto Integer = KV.second.getAsInteger())
      Metrics[KV.first.str()] = static_cast<double>(*Integer);
  }
}

static json::Object buildInspectionDiff(const json::Object &Baseline,
                                        const json::Object &Candidate) {
  json::Object Diff;
  StringMap<double> BaseMetrics;
  StringMap<double> CandidateMetrics;
  collectNumericMetrics(Baseline, BaseMetrics);
  collectNumericMetrics(Candidate, CandidateMetrics);

  StringSet<> Keys;
  for (const auto &KV : BaseMetrics)
    Keys.insert(KV.first());
  for (const auto &KV : CandidateMetrics)
    Keys.insert(KV.first());

  json::Array NumericMetrics;
  for (const auto &Key : Keys) {
    StringRef Name = Key.getKey();
    double Before = BaseMetrics.lookup(Name);
    double After = CandidateMetrics.lookup(Name);
    if (Before == After)
      continue;
    NumericMetrics.push_back(json::Object{
        {"metric", Name.str()},
        {"before", Before},
        {"after", After},
        {"delta", After - Before},
    });
  }
  Diff["numeric_metrics"] = std::move(NumericMetrics);

  json::Array ArraySizes;
  for (StringRef Key :
       {"remarks", "diagnostics", "functions", "passes", "sections"}) {
    const json::Array *Base = Baseline.getArray(Key);
    const json::Array *Cand = Candidate.getArray(Key);
    if (!Base && !Cand)
      continue;
    int64_t Before = Base ? static_cast<int64_t>(Base->size()) : 0;
    int64_t After = Cand ? static_cast<int64_t>(Cand->size()) : 0;
    if (Before == After)
      continue;
    ArraySizes.push_back(json::Object{
        {"field", Key.str()},
        {"before", Before},
        {"after", After},
        {"delta", After - Before},
    });
  }
  Diff["array_sizes"] = std::move(ArraySizes);
  return Diff;
}

CoreClient::CoreClient(std::unique_ptr<StorageManager> Storage)
    : Storage(std::move(Storage)) {
  Registry.addBuiltinRunners();
  InsightRegistry::registerBuiltinInsights();
}

Error CoreClient::loadPlugin(StringRef Path) {
  if (Error Err = Plugins.load(Path))
    return Err;
  return Plugins.registerPlugins(Registry);
}

Expected<std::unique_ptr<CoreClient>>
CoreClient::create(StringRef StoreRoot, StringRef CapabilityDir) {
  Expected<std::unique_ptr<StorageManager>> Storage =
      StorageManager::create(StoreRoot);
  if (!Storage)
    return Storage.takeError();
  std::unique_ptr<CoreClient> Client(new CoreClient(std::move(*Storage)));
  if (Error Err = Client->Registry.loadDirectory(CapabilityDir))
    return std::move(Err);
  for (const CapabilitySpec &Spec : Client->Registry.listSpecs()) {
    if (Error Err =
            Client->Storage->results().registerSchema(Spec.ID, Spec.Version))
      return std::move(Err);
  }
  return Client;
}

Expected<SnapshotRecord>
CoreClient::createSnapshot(StringRef SourceRoot, StringRef BuildRoot,
                           ArrayRef<std::string> Capabilities) {
  CaptureCore Capture(*Storage, Registry);
  return Capture.createSnapshot(SourceRoot, BuildRoot, Capabilities);
}

SmallVector<SnapshotRecord, 16> CoreClient::listSnapshots() const {
  SnapshotManager Manager(*Storage);
  return Manager.list();
}

SmallVector<UnitRecord, 64> CoreClient::listUnits(StringRef SnapshotID) const {
  return Storage->metadata().listUnits(SnapshotID);
}

SmallVector<CapabilitySpec, 32> CoreClient::listCapabilities() const {
  return Registry.listSpecs();
}

Expected<std::string> CoreClient::resolveUnitID(StringRef SnapshotID,
                                                StringRef Selector) const {
  SmallVector<UnitRecord, 64> Units = listUnits(SnapshotID);
  std::optional<std::string> SuffixMatch;
  for (const UnitRecord &Unit : Units) {
    if (Unit.ID == Selector || StringRef(Unit.ID).starts_with(Selector))
      return Unit.ID;
    if (Unit.SourcePath == Selector)
      return Unit.ID;
    if (StringRef(Unit.SourcePath).ends_with(Selector))
      SuffixMatch = Unit.ID;
  }
  if (SuffixMatch)
    return *SuffixMatch;
  return createStringError(inconvertibleErrorCode(),
                           "unknown unit in snapshot: %s",
                           Selector.str().c_str());
}

Expected<json::Array>
CoreClient::queryUnit(StringRef UnitID, ArrayRef<std::string> CapabilityIDs) {
  Expected<UnitRecord> Unit = Storage->metadata().getUnit(UnitID);
  if (!Unit)
    return Unit.takeError();

  CapabilityPlanner Planner(Registry);
  Expected<SmallVector<CapabilityNode, 16>> Plan = Planner.plan(CapabilityIDs);
  if (!Plan)
    return Plan.takeError();

  CapabilityScheduler Scheduler;
  SmallVector<CapabilityNode, 16> Schedule = Scheduler.schedule(*Plan);
  CapabilityExecutor Executor(Registry, *Storage);
  CapabilityContext Context = makeContext(*Unit);
  return Executor.execute(Schedule, Context);
}

Expected<json::Array>
CoreClient::querySnapshot(StringRef SnapshotID,
                          ArrayRef<std::string> CapabilityIDs) {
  SmallVector<UnitRecord, 64> Units = Storage->metadata().listUnits(SnapshotID);
  if (Units.empty())
    return createStringError(inconvertibleErrorCode(),
                             "snapshot has no captured units: %s",
                             SnapshotID.str().c_str());

  CapabilityPlanner Planner(Registry);
  Expected<SmallVector<CapabilityNode, 16>> Plan = Planner.plan(CapabilityIDs);
  if (!Plan)
    return Plan.takeError();

  CapabilityScheduler Scheduler;
  SmallVector<CapabilityNode, 16> Schedule = Scheduler.schedule(*Plan);
  CapabilityExecutor Executor(Registry, *Storage);
  json::Array Out;

  for (const UnitRecord &Unit : Units) {
    CapabilityContext Context = makeContext(Unit);
    Expected<json::Array> Results = Executor.execute(Schedule, Context);
    if (!Results)
      return Results.takeError();
    Out.push_back(json::Object{{"unit_id", Unit.ID},
                               {"source_path", Unit.SourcePath},
                               {"results", std::move(*Results)}});
  }

  return Out;
}

Expected<json::Object>
CoreClient::inspect(StringRef SnapshotID, StringRef UnitSelector,
                    StringRef CapabilityID,
                    const InspectionFilter &Filter) const {
  Expected<std::string> UnitID = resolveUnitID(SnapshotID, UnitSelector);
  if (!UnitID)
    return UnitID.takeError();

  Expected<UnitRecord> Unit = Storage->metadata().getUnit(*UnitID);
  if (!Unit)
    return Unit.takeError();

  Expected<json::Array> Results =
      const_cast<CoreClient *>(this)->queryUnit(*UnitID, {CapabilityID.str()});
  if (!Results)
    return Results.takeError();

  const json::Object *Value = lookupCapabilityValue(*Results, CapabilityID);
  if (!Value)
    return createStringError(inconvertibleErrorCode(),
                             "capability result missing: %s",
                             CapabilityID.str().c_str());

  json::Object Response;
  Response["snapshot_id"] = SnapshotID.str();
  Response["unit_selector"] = UnitSelector.str();
  Response["unit_id"] = *UnitID;
  Response["source_path"] = Unit->SourcePath;
  Response["capability"] = CapabilityID.str();
  Response["value"] = filterInspectionValue(*Value, Filter);
  return Response;
}

Expected<json::Object>
CoreClient::inspectSignals(StringRef SnapshotID, StringRef UnitSelector,
                           const InspectionFilter &Filter) const {
  Expected<std::string> UnitID = resolveUnitID(SnapshotID, UnitSelector);
  if (!UnitID)
    return UnitID.takeError();

  Expected<UnitRecord> Unit = Storage->metadata().getUnit(*UnitID);
  if (!Unit)
    return Unit.takeError();

  const SmallVector<std::string, 8> Capabilities = {
      "clang.diag.summary", "llvm.remarks.detail", "llvm.ir.function_stats",
      "llvm.cfg",           "llvm.dom_tree",       "llvm.loop_info",
      "llvm.debug.detail"};
  Expected<json::Array> Results = const_cast<CoreClient *>(this)->queryUnit(
      *UnitID, ArrayRef<std::string>(Capabilities));
  if (!Results)
    return Results.takeError();

  json::Array Signals;
  for (const std::string &Capability : Capabilities) {
    const json::Object *Value = lookupCapabilityValue(*Results, Capability);
    if (!Value)
      continue;
    json::Object Entry;
    Entry["capability"] = Capability;
    Entry["value"] = filterInspectionValue(*Value, Filter);
    Signals.push_back(json::Value(std::move(Entry)));
  }

  json::Object Response;
  Response["snapshot_id"] = SnapshotID.str();
  Response["unit_selector"] = UnitSelector.str();
  Response["unit_id"] = *UnitID;
  Response["source_path"] = Unit->SourcePath;
  Response["signals"] = std::move(Signals);
  return Response;
}

Expected<json::Object>
CoreClient::inspectCompare(StringRef BaselineSnapshotID, StringRef SnapshotID,
                           StringRef UnitSelector, StringRef CapabilityID,
                           const InspectionFilter &Filter) const {
  Expected<json::Object> Baseline =
      inspect(BaselineSnapshotID, UnitSelector, CapabilityID, Filter);
  if (!Baseline)
    return Baseline.takeError();
  Expected<json::Object> Candidate =
      inspect(SnapshotID, UnitSelector, CapabilityID, Filter);
  if (!Candidate)
    return Candidate.takeError();

  const json::Object *BaselineValue = Baseline->getObject("value");
  const json::Object *CandidateValue = Candidate->getObject("value");
  if (!BaselineValue || !CandidateValue)
    return createStringError(inconvertibleErrorCode(),
                             "inspection compare requires object results");

  json::Object Result;
  Result["baseline_snapshot_id"] = BaselineSnapshotID.str();
  Result["snapshot_id"] = SnapshotID.str();
  Result["unit_selector"] = UnitSelector.str();
  Result["capability"] = CapabilityID.str();
  Result["baseline"] = std::move(*Baseline);
  Result["candidate"] = std::move(*Candidate);
  Result["diff"] = buildInspectionDiff(*BaselineValue, *CandidateValue);
  return Result;
}

Expected<json::Value> CoreClient::ingestRuntime(StringRef SnapshotID,
                                                StringRef Kind,
                                                StringRef Path) {
  if (Expected<SnapshotRecord> Snapshot =
          Storage->metadata().getSnapshot(SnapshotID))
    (void)*Snapshot;
  else
    return Snapshot.takeError();

  auto Load = [&]() -> Expected<json::Value> {
    if (Kind == "pgo-instr")
      return PGOInstrIngestor().load(Path);
    if (Kind == "pgo-sample")
      return PGOSampleIngestor().load(Path);
    if (Kind == "memprof")
      return MemProfIngestor().load(Path);
    if (Kind == "coverage")
      return CoverageIngestor().load(Path);
    if (Kind == "xray")
      return XRayIngestor().load(Path);
    if (Kind == "sanitizer")
      return SanitizerIngestor().load(Path);
    if (Kind == "sancov")
      return SancovIngestor().load(Path);
    if (Kind == "offload")
      return OffloadRuntime().load(Path);
    return createStringError(inconvertibleErrorCode(),
                             "unknown runtime kind: %s", Kind.str().c_str());
  };
  Expected<json::Value> Loaded = Load();
  if (!Loaded)
    return Loaded.takeError();

  Expected<std::string> BlobID = Storage->blobs().put(stringifyJSON(*Loaded));
  if (!BlobID)
    return BlobID.takeError();

  const json::Object *Object = Loaded->getAsObject();
  if (!Object)
    return createStringError(inconvertibleErrorCode(),
                             "runtime ingestor returned non-object payload");
  std::string RuntimeKind = Kind.str();
  if (std::optional<StringRef> PayloadKind = Object->getString("kind"))
    RuntimeKind = PayloadKind->str();

  EntityRecord Representation;
  Representation.Kind = "representation";
  Representation.ID =
      "repr_" + hashString((SnapshotID + RuntimeKind + *BlobID).str());
  Representation.SnapshotID = SnapshotID.str();
  Representation.OwnerID = SnapshotID.str();
  Representation.Data = *Object;
  Representation.Data["representation_id"] = Representation.ID;
  Representation.Data["owner_id"] = SnapshotID.str();
  Representation.Data["owner_type"] = "snapshot";
  Representation.Data["capability_id"] = ("runtime." + Kind).str();
  Representation.Data["capability_version"] = "1";
  Representation.Data["content_type"] = "application/json";
  Representation.Data["content_address"] = *BlobID;
  Representation.Data["materialization_policy"] = "eager";
  Representation.Data["readiness_level"] = "L3";
  if (Error Err = Storage->metadata().putEntity(Representation))
    return std::move(Err);

  if (const json::Array *Findings = Object->getArray("findings")) {
    for (const json::Value &FindingValue : *Findings) {
      const json::Object *FindingObject = FindingValue.getAsObject();
      if (!FindingObject)
        continue;
      EntityRecord Finding;
      Finding.Kind = "finding";
      Finding.ID = "finding_" +
                   hashString(Representation.ID + stringifyJSON(FindingValue));
      Finding.SnapshotID = SnapshotID.str();
      Finding.OwnerID = Representation.ID;
      Finding.Data = *FindingObject;
      Finding.Data["finding_id"] = Finding.ID;
      Finding.Data["owner_id"] = Representation.ID;
      Finding.Data["owner_type"] = "snapshot";
      Finding.Data["representation_id"] = Representation.ID;
      if (Error Err = Storage->metadata().putEntity(Finding))
        return std::move(Err);
    }
  }

  json::Object Response = *Object;
  Response["representation_id"] = Representation.ID;
  Response["content_address"] = *BlobID;
  return Response;
}

Expected<json::Value> CoreClient::correlateRuntime(StringRef SnapshotID) {
  RuntimeCorrelator Correlator;
  return Correlator.correlate(*Storage, SnapshotID);
}

// Fetch capability data for a unit. Returns the parsed json::Object from the
// result value, or an error if the capability is unavailable.
static Expected<json::Value> fetchCapabilityData(CoreClient &Client,
                                                 StringRef UnitID,
                                                 StringRef Capability) {
  Expected<json::Array> R = Client.queryUnit(UnitID, {Capability.str()});
  if (!R)
    return R.takeError();
  const json::Object *Value = lookupCapabilityValue(*R, Capability);
  if (!Value)
    return createStringError(inconvertibleErrorCode(),
                             "missing capability value for: %s",
                             Capability.str().c_str());
  if (Value->getBoolean("available") == false)
    return createStringError(inconvertibleErrorCode(), "%s",
                             Value->getString("reason")
                                 .value_or("capability unavailable")
                                 .str()
                                 .c_str());
  json::Object Copy = *Value;
  return json::Value(std::move(Copy));
}

// Pick the first unit from a snapshot, or return an error.
static Expected<std::string>
resolveUnit(CoreClient &Client, StringRef SnapshotID, StringRef UnitID) {
  if (!UnitID.empty())
    return UnitID.str();
  SmallVector<UnitRecord, 64> Units = Client.listUnits(SnapshotID);
  if (Units.empty())
    return createStringError(inconvertibleErrorCode(),
                             "snapshot has no captured units: %s",
                             SnapshotID.str().c_str());
  return Units.front().ID;
}

Expected<json::Array> CoreClient::listInsights(StringRef SnapshotID,
                                               StringRef UnitID) {
  if (SnapshotID.empty())
    return createStringError(inconvertibleErrorCode(),
                             "insight-list requires a snapshot id");

  Expected<std::string> ChosenUnit = resolveUnit(*this, SnapshotID, UnitID);
  if (!ChosenUnit)
    return ChosenUnit.takeError();

  SmallVector<Insight *, 16> All = InsightRegistry::instance().all();
  json::Array Out;

  for (const Insight *I : All) {
    StringRef Cap = I->getRequiredCapability();
    Expected<json::Value> Data = fetchCapabilityData(*this, *ChosenUnit, Cap);
    bool Available = static_cast<bool>(Data);
    std::string Reason;
    if (!Available)
      Reason = toString(Data.takeError());

    Out.push_back(json::Object{
        {"name", I->getName()},
        {"description", I->getDescription()},
        {"required_capability", Cap},
        {"requires_baseline", I->requiresBaseline()},
        {"available", Available},
        {"reason", Reason},
    });
  }
  return Out;
}

Expected<json::Object> CoreClient::runInsight(StringRef Name,
                                              StringRef SnapshotID,
                                              StringRef UnitID,
                                              StringRef BaselineSnapshotID) {
  if (SnapshotID.empty())
    return createStringError(inconvertibleErrorCode(),
                             "runInsight requires a snapshot id");

  Insight *I = InsightRegistry::instance().get(Name);
  if (!I)
    return createStringError(inconvertibleErrorCode(), "unknown insight: %s",
                             Name.str().c_str());

  Expected<std::string> ChosenUnit = resolveUnit(*this, SnapshotID, UnitID);
  if (!ChosenUnit)
    return ChosenUnit.takeError();

  // Fetch primary capability data.
  StringRef Cap = I->getRequiredCapability();
  Expected<json::Value> PrimaryData =
      fetchCapabilityData(*this, *ChosenUnit, Cap);
  if (!PrimaryData)
    return PrimaryData.takeError();

  InsightInput Input;
  Input.UnitId = *ChosenUnit;
  Input.SnapshotId = SnapshotID.str();

  const json::Object *PrimaryObj = PrimaryData->getAsObject();
  if (!PrimaryObj)
    return createStringError(inconvertibleErrorCode(),
                             "capability data is not a JSON object: %s",
                             Cap.str().c_str());
  Input.PrimaryData = PrimaryObj;

  // Fetch baseline data if the insight requires it.
  std::optional<json::Value> BaselineStorage;
  if (I->requiresBaseline()) {
    if (BaselineSnapshotID.empty())
      return createStringError(inconvertibleErrorCode(),
                               "insight '%s' requires --baseline snapshot",
                               Name.str().c_str());

    Expected<std::string> BaseUnit =
        resolveUnit(*this, BaselineSnapshotID, UnitID);
    if (!BaseUnit)
      return BaseUnit.takeError();

    Expected<json::Value> BaseData = fetchCapabilityData(*this, *BaseUnit, Cap);
    if (!BaseData)
      return BaseData.takeError();

    BaselineStorage = std::move(*BaseData);
    Input.BaselineSnapshotId = BaselineSnapshotID.str();
    Input.BaselineData = BaselineStorage->getAsObject();
    if (!Input.BaselineData)
      return createStringError(
          inconvertibleErrorCode(),
          "baseline capability data is not a JSON object: %s",
          Cap.str().c_str());
  }

  Expected<InsightOutput> Result = I->analyze(Input);
  if (!Result)
    return Result.takeError();

  json::Object Out;
  Out["insight"] = Name.str();
  Out["snapshot_id"] = SnapshotID.str();
  Out["unit_id"] = *ChosenUnit;
  if (!BaselineSnapshotID.empty())
    Out["baseline_snapshot_id"] = BaselineSnapshotID.str();
  Out["data"] = std::move(Result->Data);

  json::Array Warnings;
  for (auto &W : Result->Warnings)
    Warnings.push_back(W);
  if (!Warnings.empty())
    Out["warnings"] = std::move(Warnings);

  return Out;
}

SmallVector<JobRecord, 16> CoreClient::listJobs() const {
  return Storage->metadata().listJobs();
}

Expected<JobRecord> CoreClient::getJob(StringRef JobID) const {
  return Storage->metadata().getJob(JobID);
}

json::Value CoreClient::compare(StringRef Before, StringRef After) const {
  CompareEngine Engine(*Storage);
  return Engine.compare(Before, After);
}

json::Value CoreClient::compareCapability(StringRef Before, StringRef After,
                                          StringRef CapID) const {
  CompareEngine Engine(*Storage);
  return Engine.compareCapability(Before, After, CapID);
}

HealthStatus CoreClient::health() const {
  HealthStatus Status;
  Status.OK = true;
  Status.Store = Storage->root().str();
  Status.Snapshots = Storage->metadata().snapshotCount();
  Status.Units = Storage->metadata().unitCount();
  return Status;
}

json::Value CoreClient::inspectStorage() const {
  return json::Object{
      {"root", Storage->root()},
      {"snapshots", static_cast<int64_t>(Storage->metadata().snapshotCount())},
      {"units", static_cast<int64_t>(Storage->metadata().unitCount())},
      {"schema_version",
       static_cast<int64_t>(Storage->schema().getCurrentVersion())}};
}

Error CoreClient::compactStorage() { return Storage->retention().compact(); }
