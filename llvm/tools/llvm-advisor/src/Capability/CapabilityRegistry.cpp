//===------------------- CapabilityRegistry.cpp - LLVM Advisor
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

#include "Capability/CapabilityRegistry.h"
#include "Analysis/Binary/CGDataAnalyzer.h"
#include "Analysis/Binary/DebugConsistencyAnalyzer.h"
#include "Analysis/Binary/DebugDetailAnalyzer.h"
#include "Analysis/Binary/DebugSummaryAnalyzer.h"
#include "Analysis/Binary/LinkerMapAnalyzer.h"
#include "Analysis/Binary/ObjectAnalyzer.h"
#include "Analysis/Binary/ReadObjAnalyzer.h"
#include "Analysis/Build/BuildMetaAnalyzer.h"
#include "Analysis/Build/HeaderDepsAnalyzer.h"
#include "Analysis/Build/MacroExpansionAnalyzer.h"
#include "Analysis/Build/ModulesAnalyzer.h"
#include "Analysis/Build/PreprocessorAnalyzer.h"
#include "Analysis/Build/TemplateStatsAnalyzer.h"
#include "Analysis/Build/TimeTraceAnalyzer.h"
#include "Analysis/Clang/ASTAnalyzer.h"
#include "Analysis/Clang/ASTJsonAnalyzer.h"
#include "Analysis/Clang/DiagnosticsAnalyzer.h"
#include "Analysis/Clang/SARIFDiagnosticsAnalyzer.h"
#include "Analysis/Clang/StaticAnalysisAnalyzer.h"
#include "Analysis/IR/BitcodeAnalyzer.h"
#include "Analysis/IR/FunctionStatsAnalyzer.h"
#include "Analysis/IR/IRAnalyzer.h"
#include "Analysis/IR/InliningTreeAnalyzer.h"
#include "Analysis/IR/PassStatsAnalyzer.h"
#include "Analysis/IR/PassTimingAnalyzer.h"
#include "Analysis/IR/RemarksAnalyzer.h"
#include "Analysis/IR/RemarksMixAnalyzer.h"
#include "Analysis/IR/RemarksSizeDiffAnalyzer.h"
#include "Analysis/IR/SimilarityAnalyzer.h"
#include "Analysis/Inspection/AsmViewAnalyzer.h"
#include "Analysis/Inspection/CFGAnalyzer.h"
#include "Analysis/Inspection/CallGraphAnalyzer.h"
#include "Analysis/Inspection/DomTreeAnalyzer.h"
#include "Analysis/Inspection/ExegesisAnalyzer.h"
#include "Analysis/Inspection/IRDiffAnalyzer.h"
#include "Analysis/Inspection/IRViewAnalyzer.h"
#include "Analysis/Inspection/LoopInfoAnalyzer.h"
#include "Analysis/Inspection/MCAAnalyzer.h"
#include "Analysis/Inspection/MachineIRAnalyzer.h"
#include "Analysis/Inspection/PassListAnalyzer.h"
#include "Analysis/Inspection/RemarksDetailAnalyzer.h"
#include "Analysis/Inspection/SelectionDAGAnalyzer.h"
#include "Analysis/LTO/LTOAnalyzer.h"
#include "Analysis/LTO/LTOFunctionStatsAnalyzer.h"
#include "Analysis/LTO/OffloadBinaryAnalyzer.h"
#include "Analysis/Offload/DeviceProfileAnalyzer.h"
#include "Analysis/Offload/DeviceTraceAnalyzer.h"
#include "Analysis/Offload/HostRegionsAnalyzer.h"
#include "Analysis/Offload/MemoryProfileAnalyzer.h"
#include "Analysis/Offload/MemoryTransferAnalyzer.h"
#include "Analysis/Offload/OffloadAnalyzer.h"
#include "Analysis/Offload/SyncAnalysisAnalyzer.h"
#include "Analysis/Offload/SyncPointsAnalyzer.h"
#include "Analysis/Runtime/RuntimeRunner.h"
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
    return createStringError(EC, "cannot read capability directory '%s'",
                             ConfigDir.str().c_str());
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
    return createStringError(inconvertibleErrorCode(), "unknown capability: %s",
                             ID.str().c_str());
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
  consumeError(
      addRunner("builtin.build_meta", std::make_unique<BuildMetaAnalyzer>()));
  consumeError(
      addRunner("builtin.header_deps", std::make_unique<HeaderDepsAnalyzer>()));
  consumeError(
      addRunner("builtin.time_trace", std::make_unique<TimeTraceAnalyzer>()));
  consumeError(addRunner("builtin.template_stats",
                         std::make_unique<TemplateStatsAnalyzer>()));
  consumeError(addRunner("builtin.preprocessor",
                         std::make_unique<PreprocessorAnalyzer>()));
  consumeError(addRunner("builtin.macro_expansion",
                         std::make_unique<MacroExpansionAnalyzer>()));
  consumeError(
      addRunner("builtin.modules", std::make_unique<ModulesAnalyzer>()));
  consumeError(addRunner("builtin.clang_diag_summary",
                         std::make_unique<DiagnosticsAnalyzer>()));
  consumeError(addRunner("builtin.clang_diag_sarif",
                         std::make_unique<SARIFDiagnosticsAnalyzer>()));
  consumeError(
      addRunner("builtin.clang_ast_summary", std::make_unique<ASTAnalyzer>()));
  consumeError(
      addRunner("builtin.clang_ast_json", std::make_unique<ASTJsonAnalyzer>()));
  consumeError(addRunner("builtin.clang_static_analysis",
                         std::make_unique<StaticAnalysisAnalyzer>()));
  consumeError(addRunner("builtin.ir_summary", std::make_unique<IRAnalyzer>()));
  consumeError(addRunner("builtin.ir_function_stats",
                         std::make_unique<FunctionStatsAnalyzer>()));
  consumeError(
      addRunner("builtin.object_summary", std::make_unique<ObjectAnalyzer>()));

  // IR analyzers
  consumeError(addRunner("builtin.inlining_tree",
                         std::make_unique<InliningTreeAnalyzer>()));
  consumeError(addRunner("builtin.remarks_summary",
                         std::make_unique<RemarksAnalyzer>()));
  consumeError(
      addRunner("builtin.pass_stats", std::make_unique<PassStatsAnalyzer>()));
  consumeError(
      addRunner("builtin.pass_timing", std::make_unique<PassTimingAnalyzer>()));
  consumeError(addRunner("builtin.ir_similarity",
                         std::make_unique<SimilarityAnalyzer>()));
  consumeError(
      addRunner("builtin.bcanalyzer", std::make_unique<BitcodeAnalyzer>()));
  consumeError(
      addRunner("builtin.remarks_mix", std::make_unique<RemarksMixAnalyzer>()));
  consumeError(addRunner("builtin.remarks_size_diff",
                         std::make_unique<RemarksSizeDiffAnalyzer>()));

  // Inspection analyzers
  consumeError(
      addRunner("builtin.passes_list", std::make_unique<PassListAnalyzer>()));
  consumeError(
      addRunner("builtin.ir_view", std::make_unique<IRViewAnalyzer>()));
  consumeError(
      addRunner("builtin.ir_diff", std::make_unique<IRDiffAnalyzer>()));
  consumeError(addRunner("builtin.cfg", std::make_unique<CFGAnalyzer>()));
  consumeError(
      addRunner("builtin.dom_tree", std::make_unique<DomTreeAnalyzer>()));
  consumeError(
      addRunner("builtin.call_graph", std::make_unique<CallGraphAnalyzer>()));
  consumeError(
      addRunner("builtin.loop_info", std::make_unique<LoopInfoAnalyzer>()));
  consumeError(addRunner("builtin.selection_dag",
                         std::make_unique<SelectionDAGAnalyzer>()));
  consumeError(
      addRunner("builtin.machine_ir", std::make_unique<MachineIRAnalyzer>()));
  consumeError(
      addRunner("builtin.asm_view", std::make_unique<AsmViewAnalyzer>()));
  consumeError(addRunner("builtin.remarks_detail",
                         std::make_unique<RemarksDetailAnalyzer>()));
  consumeError(
      addRunner("builtin.mca_report", std::make_unique<MCAAnalyzer>()));
  consumeError(
      addRunner("builtin.exegesis", std::make_unique<ExegesisAnalyzer>()));

  // Binary analyzers
  consumeError(
      addRunner("builtin.readobj", std::make_unique<ReadObjAnalyzer>()));
  consumeError(addRunner("builtin.debug_summary",
                         std::make_unique<DebugSummaryAnalyzer>()));
  consumeError(addRunner("builtin.debug_detail",
                         std::make_unique<DebugDetailAnalyzer>()));
  consumeError(addRunner("builtin.debug_consistency",
                         std::make_unique<DebugConsistencyAnalyzer>()));
  consumeError(addRunner("builtin.cgdata", std::make_unique<CGDataAnalyzer>()));
  consumeError(
      addRunner("builtin.linker_map", std::make_unique<LinkerMapAnalyzer>()));

  // LTO analyzers
  consumeError(
      addRunner("builtin.lto_summary", std::make_unique<LTOAnalyzer>()));
  consumeError(addRunner("builtin.lto_function_stats",
                         std::make_unique<LTOFunctionStatsAnalyzer>()));
  consumeError(addRunner("builtin.offload_binary",
                         std::make_unique<OffloadBinaryAnalyzer>()));

  // Offload analyzers
  consumeError(addRunner("builtin.offload_summary",
                         std::make_unique<OffloadAnalyzer>()));
  consumeError(addRunner("builtin.device_profile",
                         std::make_unique<DeviceProfileAnalyzer>()));
  consumeError(addRunner("builtin.memory_transfer",
                         std::make_unique<MemoryTransferAnalyzer>()));
  consumeError(
      addRunner("builtin.sync_points", std::make_unique<SyncPointsAnalyzer>()));
  consumeError(addRunner("builtin.host_regions",
                         std::make_unique<HostRegionsAnalyzer>()));
  consumeError(addRunner("builtin.device_trace",
                         std::make_unique<DeviceTraceAnalyzer>()));
  consumeError(addRunner("builtin.memory_profile",
                         std::make_unique<MemoryProfileAnalyzer>()));
  consumeError(addRunner("builtin.sync_analysis",
                         std::make_unique<SyncAnalysisAnalyzer>()));

  // Runtime analyzers
  consumeError(
      addRunner("builtin.pgo_instr", std::make_unique<PGOInstrRunner>()));
  consumeError(
      addRunner("builtin.pgo_sample", std::make_unique<PGOSampleRunner>()));
  consumeError(addRunner("builtin.memprof", std::make_unique<MemProfRunner>()));
  consumeError(
      addRunner("builtin.coverage", std::make_unique<CoverageRunner>()));
  consumeError(addRunner("builtin.xray", std::make_unique<XRayRunner>()));
  consumeError(addRunner("builtin.sancov", std::make_unique<SancovRunner>()));
  consumeError(addRunner(
      "builtin.sanitizer_asan",
      std::make_unique<SanitizerRunner>("runtime.sanitizer.asan", "asan")));
  consumeError(addRunner(
      "builtin.sanitizer_ubsan",
      std::make_unique<SanitizerRunner>("runtime.sanitizer.ubsan", "ubsan")));
  consumeError(addRunner(
      "builtin.sanitizer_msan",
      std::make_unique<SanitizerRunner>("runtime.sanitizer.msan", "msan")));
  consumeError(addRunner(
      "builtin.sanitizer_tsan",
      std::make_unique<SanitizerRunner>("runtime.sanitizer.tsan", "tsan")));
}
