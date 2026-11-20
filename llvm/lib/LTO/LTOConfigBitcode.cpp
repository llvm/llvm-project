//===- LTOConfigBitcode.cpp - lto::Config in bitcode ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes and decodes the lto::Config JSON blob in ThinLTO summary indexes.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"

#include "JSONConfigUtils.h"

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;

void encodePipelineTuningOptions(JSONWriter &Writer,
                                 const PipelineTuningOptions &PTO) {
  [[maybe_unused]] const auto &[LoopInterleaving, LoopVectorization,
                                SLPVectorization, LoopUnrolling,
                                LoopInterchange, LoopFusion,
                                ForgetAllSCEVInLoopUnroll, LicmMssaOptCap,
                                LicmMssaNoAccForPromotionCap, CallGraphProfile,
                                UnifiedLTO, MergeFunctions, InlinerThreshold,
                                MaxDevirtIterations, EagerlyInvalidateAnalyses,
                                DevirtualizeSpeculatively] = PTO;

  Writer.putBool("pto.LoopInterleaving", PTO.LoopInterleaving);
  Writer.putBool("pto.LoopVectorization", PTO.LoopVectorization);
  Writer.putBool("pto.SLPVectorization", PTO.SLPVectorization);
  Writer.putBool("pto.LoopUnrolling", PTO.LoopUnrolling);
  Writer.putBool("pto.LoopInterchange", PTO.LoopInterchange);
  Writer.putBool("pto.LoopFusion", PTO.LoopFusion);
  if (PTO.ForgetAllSCEVInLoopUnroll)
    Writer.putBool("pto.ForgetAllSCEVInLoopUnroll",
                   *PTO.ForgetAllSCEVInLoopUnroll);
  if (PTO.LicmMssaOptCap)
    Writer.putU32("pto.LicmMssaOptCap", *PTO.LicmMssaOptCap);
  if (PTO.LicmMssaNoAccForPromotionCap)
    Writer.putU32("pto.LicmMssaNoAccForPromotionCap",
                  *PTO.LicmMssaNoAccForPromotionCap);
  Writer.putBool("pto.CallGraphProfile", PTO.CallGraphProfile);
  Writer.putBool("pto.UnifiedLTO", PTO.UnifiedLTO);
  if (PTO.MergeFunctions)
    Writer.putBool("pto.MergeFunctions", *PTO.MergeFunctions);
  Writer.putI32("pto.InlinerThreshold", PTO.InlinerThreshold);
  if (PTO.MaxDevirtIterations)
    Writer.putU32("pto.MaxDevirtIterations", *PTO.MaxDevirtIterations);
  if (PTO.EagerlyInvalidateAnalyses)
    Writer.putBool("pto.EagerlyInvalidateAnalyses",
                   *PTO.EagerlyInvalidateAnalyses);
  if (PTO.DevirtualizeSpeculatively)
    Writer.putBool("pto.DevirtualizeSpeculatively",
                   *PTO.DevirtualizeSpeculatively);
}

void encodeConfigFields(JSONWriter &Writer, const Config &C) {
  // Keep this decomposition in sync with Config. It intentionally includes
  // non-serializable fields so that adding or removing any field produces a
  // compile error here. New fields must either be added to encodeConfigFields
  // and applyEntry or explicitly documented as non-serializable below.
  [[maybe_unused]] const auto
      &[LibraryOptions, // Process-local pointer.
        CPU, Options, MAttrs, MllvmArgs,
        LoadedPassPlugins, // Non-serializable process-local pointers.
        PassPluginFilenames,
        PassBuilderCallback, // Non-serializable callback.
        RelocModel, CodeModel, CGOptLevel, CGFileType, OptLevel, VerifyEach,
        DisableVerify, Freestanding, CodeGenOnly, RunCSIRInstr, PGOWarnMismatch,
        HasWholeProgramVisibility, ValidateAllVtablesHaveTypeInfos,
        AllVtablesHaveTypeInfos, AlwaysEmitRegularLTOObj, KeepSymbolNameCopies,
        Dtlto, VisibilityScheme, OptPipeline, AAPipeline, OverrideTriple,
        DefaultTriple, CSIRProfile, SampleProfile, ProfileRemapping, DwoDir,
        SplitDwarfFile, SplitDwarfOutput, RemarksFilename, RemarksPasses,
        RemarksWithHotness, RemarksHotnessThreshold, RemarksFormat,
        DebugPassManager, StatsFile, ThinLTOModulesToCompile, TimeTraceEnabled,
        TimeTraceGranularity, ShouldDiscardValueNames,
        DiagHandler, // Non-serializable callback.
        AddFSDiscriminator,
        ResolutionFile, // Non-serializable stream.
        PTO,
        PreOptModuleHook, // Non-serializable callbacks.
        PostPromoteModuleHook, PostInternalizeModuleHook, PostImportModuleHook,
        PostOptModuleHook, PreCodeGenModuleHook, CombinedIndexHook,
        PostRegularLTOLinkHook, GetSummaryIndexOutputStream,
        GetImportsListOutputArray, GetCacheKeyOutputString] = C;

  Writer.putString("CPU", C.CPU);
  Writer.putValue("Options", encodeTargetOptionsJSON(C.Options));
  Writer.putStringList("MAttrs", C.MAttrs);
  Writer.putStringList("MllvmArgs", C.MllvmArgs);
  Writer.putStringList("PassPluginFilenames", C.PassPluginFilenames);

  Writer.putValue("RelocModel",
                  C.RelocModel ? json::Value(static_cast<int>(*C.RelocModel))
                               : json::Value(nullptr));
  if (C.CodeModel)
    Writer.putI32("CodeModel", static_cast<int32_t>(*C.CodeModel));

  Writer.putI32("CGOptLevel", static_cast<int32_t>(C.CGOptLevel));
  Writer.putI32("CGFileType", static_cast<int32_t>(C.CGFileType));
  Writer.putU32("OptLevel", C.OptLevel);

  Writer.putBool("VerifyEach", C.VerifyEach);
  Writer.putBool("DisableVerify", C.DisableVerify);
  Writer.putBool("Freestanding", C.Freestanding);
  Writer.putBool("CodeGenOnly", C.CodeGenOnly);
  Writer.putBool("RunCSIRInstr", C.RunCSIRInstr);
  Writer.putBool("PGOWarnMismatch", C.PGOWarnMismatch);
  Writer.putBool("HasWholeProgramVisibility", C.HasWholeProgramVisibility);
  Writer.putBool("ValidateAllVtablesHaveTypeInfos",
                 C.ValidateAllVtablesHaveTypeInfos);
  Writer.putBool("AllVtablesHaveTypeInfos", C.AllVtablesHaveTypeInfos);
  Writer.putBool("AlwaysEmitRegularLTOObj", C.AlwaysEmitRegularLTOObj);
  Writer.putBool("KeepSymbolNameCopies", C.KeepSymbolNameCopies);
  Writer.putBool("Dtlto", C.Dtlto);
  Writer.putI32("VisibilityScheme", static_cast<int32_t>(C.VisibilityScheme));

  Writer.putString("OptPipeline", C.OptPipeline);
  Writer.putString("AAPipeline", C.AAPipeline);
  Writer.putString("OverrideTriple", C.OverrideTriple);
  Writer.putString("DefaultTriple", C.DefaultTriple);
  Writer.putString("CSIRProfile", C.CSIRProfile);
  Writer.putString("SampleProfile", C.SampleProfile);
  Writer.putString("ProfileRemapping", C.ProfileRemapping);
  Writer.putString("DwoDir", C.DwoDir);
  Writer.putString("SplitDwarfFile", C.SplitDwarfFile);
  Writer.putString("SplitDwarfOutput", C.SplitDwarfOutput);
  Writer.putString("RemarksFilename", C.RemarksFilename);
  Writer.putString("RemarksPasses", C.RemarksPasses);
  Writer.putBool("RemarksWithHotness", C.RemarksWithHotness);
  Writer.putValue("RemarksHotnessThreshold",
                  json::toJSON(C.RemarksHotnessThreshold));
  Writer.putString("RemarksFormat", C.RemarksFormat);
  Writer.putBool("DebugPassManager", C.DebugPassManager);
  Writer.putString("StatsFile", C.StatsFile);
  Writer.putStringList("ThinLTOModulesToCompile", C.ThinLTOModulesToCompile);
  Writer.putBool("TimeTraceEnabled", C.TimeTraceEnabled);
  Writer.putU32("TimeTraceGranularity", C.TimeTraceGranularity);
  Writer.putBool("ShouldDiscardValueNames", C.ShouldDiscardValueNames);
  Writer.putBool("AddFSDiscriminator", C.AddFSDiscriminator);

  encodePipelineTuningOptions(Writer, C.PTO);
}

Error decodeRelocModel(Config &C, JSONEntryApplier Applier) {
  if (Applier.Entry.getAsNull()) {
    C.RelocModel.reset();
    return Error::success();
  }
  return Applier.applyI32(
      [&](int32_t V) { C.RelocModel = static_cast<Reloc::Model>(V); });
}

Expected<TargetOptions> decodeOptions(const json::Value &Entry) {
  return decodeTargetOptionsJSON(Entry);
}

Error decodeRemarksHotnessThreshold(Config &C, const json::Value &Entry) {
  if (Entry.getAsNull()) {
    C.RemarksHotnessThreshold.reset();
    return Error::success();
  }
  auto Value = Entry.getAsUINT64();
  if (!Value)
    return configError(
        "RemarksHotnessThreshold: expected an unsigned integer or null");
  C.RemarksHotnessThreshold = *Value;
  return Error::success();
}

Error applyEntry(Config &C, StringRef Key, JSONEntryApplier Applier) {
  if (Key == "CPU")
    return Applier.applyString([&](StringRef V) { C.CPU = V.str(); });
  if (Key == "Options") {
    auto Opt = decodeOptions(Applier.Entry);
    if (!Opt)
      return Opt.takeError();
    C.Options = std::move(*Opt);
    return Error::success();
  }
  if (Key == "MAttrs")
    return Applier.applyStringList(
        [&](std::vector<std::string> V) { C.MAttrs = std::move(V); });
  if (Key == "MllvmArgs")
    return Applier.applyStringList(
        [&](std::vector<std::string> V) { C.MllvmArgs = std::move(V); });
  if (Key == "PassPluginFilenames")
    return Applier.applyStringList([&](std::vector<std::string> V) {
      C.PassPluginFilenames = std::move(V);
    });
  if (Key == "RelocModel")
    return decodeRelocModel(C, Applier);
  if (Key == "CodeModel")
    return Applier.applyI32(
        [&](int32_t V) { C.CodeModel = static_cast<CodeModel::Model>(V); });
  if (Key == "CGOptLevel")
    return Applier.applyI32(
        [&](int32_t V) { C.CGOptLevel = static_cast<CodeGenOptLevel>(V); });
  if (Key == "CGFileType")
    return Applier.applyI32(
        [&](int32_t V) { C.CGFileType = static_cast<CodeGenFileType>(V); });
  if (Key == "OptLevel")
    return Applier.applyU32([&](uint32_t V) { C.OptLevel = V; });

  if (Key == "VerifyEach")
    return Applier.applyBool([&](bool V) { C.VerifyEach = V; });
  if (Key == "DisableVerify")
    return Applier.applyBool([&](bool V) { C.DisableVerify = V; });
  if (Key == "Freestanding")
    return Applier.applyBool([&](bool V) { C.Freestanding = V; });
  if (Key == "CodeGenOnly")
    return Applier.applyBool([&](bool V) { C.CodeGenOnly = V; });
  if (Key == "RunCSIRInstr")
    return Applier.applyBool([&](bool V) { C.RunCSIRInstr = V; });
  if (Key == "PGOWarnMismatch")
    return Applier.applyBool([&](bool V) { C.PGOWarnMismatch = V; });
  if (Key == "HasWholeProgramVisibility")
    return Applier.applyBool([&](bool V) { C.HasWholeProgramVisibility = V; });
  if (Key == "ValidateAllVtablesHaveTypeInfos")
    return Applier.applyBool(
        [&](bool V) { C.ValidateAllVtablesHaveTypeInfos = V; });
  if (Key == "AllVtablesHaveTypeInfos")
    return Applier.applyBool([&](bool V) { C.AllVtablesHaveTypeInfos = V; });
  if (Key == "AlwaysEmitRegularLTOObj")
    return Applier.applyBool([&](bool V) { C.AlwaysEmitRegularLTOObj = V; });
  if (Key == "KeepSymbolNameCopies")
    return Applier.applyBool([&](bool V) { C.KeepSymbolNameCopies = V; });
  if (Key == "Dtlto")
    return Applier.applyBool([&](bool V) { C.Dtlto = V; });
  if (Key == "VisibilityScheme")
    return Applier.applyI32([&](int32_t V) {
      C.VisibilityScheme = static_cast<Config::VisScheme>(V);
    });

  if (Key == "OptPipeline")
    return Applier.applyString([&](StringRef V) { C.OptPipeline = V.str(); });
  if (Key == "AAPipeline")
    return Applier.applyString([&](StringRef V) { C.AAPipeline = V.str(); });
  if (Key == "OverrideTriple")
    return Applier.applyString(
        [&](StringRef V) { C.OverrideTriple = V.str(); });
  if (Key == "DefaultTriple")
    return Applier.applyString([&](StringRef V) { C.DefaultTriple = V.str(); });
  if (Key == "CSIRProfile")
    return Applier.applyString([&](StringRef V) { C.CSIRProfile = V.str(); });
  if (Key == "SampleProfile")
    return Applier.applyString([&](StringRef V) { C.SampleProfile = V.str(); });
  if (Key == "ProfileRemapping")
    return Applier.applyString(
        [&](StringRef V) { C.ProfileRemapping = V.str(); });
  if (Key == "DwoDir")
    return Applier.applyString([&](StringRef V) { C.DwoDir = V.str(); });
  if (Key == "SplitDwarfFile")
    return Applier.applyString(
        [&](StringRef V) { C.SplitDwarfFile = V.str(); });
  if (Key == "SplitDwarfOutput")
    return Applier.applyString(
        [&](StringRef V) { C.SplitDwarfOutput = V.str(); });
  if (Key == "RemarksFilename")
    return Applier.applyString(
        [&](StringRef V) { C.RemarksFilename = V.str(); });
  if (Key == "RemarksPasses")
    return Applier.applyString([&](StringRef V) { C.RemarksPasses = V.str(); });
  if (Key == "RemarksWithHotness")
    return Applier.applyBool([&](bool V) { C.RemarksWithHotness = V; });
  if (Key == "RemarksHotnessThreshold")
    return decodeRemarksHotnessThreshold(C, Applier.Entry);
  if (Key == "RemarksFormat")
    return Applier.applyString([&](StringRef V) { C.RemarksFormat = V.str(); });
  if (Key == "DebugPassManager")
    return Applier.applyBool([&](bool V) { C.DebugPassManager = V; });
  if (Key == "StatsFile")
    return Applier.applyString([&](StringRef V) { C.StatsFile = V.str(); });
  if (Key == "ThinLTOModulesToCompile")
    return Applier.applyStringList([&](std::vector<std::string> V) {
      C.ThinLTOModulesToCompile = std::move(V);
    });
  if (Key == "TimeTraceEnabled")
    return Applier.applyBool([&](bool V) { C.TimeTraceEnabled = V; });
  if (Key == "TimeTraceGranularity")
    return Applier.applyU32([&](uint32_t V) { C.TimeTraceGranularity = V; });
  if (Key == "ShouldDiscardValueNames")
    return Applier.applyBool([&](bool V) { C.ShouldDiscardValueNames = V; });
  if (Key == "AddFSDiscriminator")
    return Applier.applyBool([&](bool V) { C.AddFSDiscriminator = V; });

  PipelineTuningOptions &PTO = C.PTO;
  if (Key == "pto.LoopInterleaving")
    return Applier.applyBool([&](bool V) { PTO.LoopInterleaving = V; });
  if (Key == "pto.LoopVectorization")
    return Applier.applyBool([&](bool V) { PTO.LoopVectorization = V; });
  if (Key == "pto.SLPVectorization")
    return Applier.applyBool([&](bool V) { PTO.SLPVectorization = V; });
  if (Key == "pto.LoopUnrolling")
    return Applier.applyBool([&](bool V) { PTO.LoopUnrolling = V; });
  if (Key == "pto.LoopInterchange")
    return Applier.applyBool([&](bool V) { PTO.LoopInterchange = V; });
  if (Key == "pto.LoopFusion")
    return Applier.applyBool([&](bool V) { PTO.LoopFusion = V; });
  if (Key == "pto.ForgetAllSCEVInLoopUnroll")
    return Applier.applyBool(
        [&](bool V) { PTO.ForgetAllSCEVInLoopUnroll = V; });
  if (Key == "pto.LicmMssaOptCap")
    return Applier.applyU32([&](uint32_t V) { PTO.LicmMssaOptCap = V; });
  if (Key == "pto.LicmMssaNoAccForPromotionCap")
    return Applier.applyU32(
        [&](uint32_t V) { PTO.LicmMssaNoAccForPromotionCap = V; });
  if (Key == "pto.CallGraphProfile")
    return Applier.applyBool([&](bool V) { PTO.CallGraphProfile = V; });
  if (Key == "pto.UnifiedLTO")
    return Applier.applyBool([&](bool V) { PTO.UnifiedLTO = V; });
  if (Key == "pto.MergeFunctions")
    return Applier.applyBool([&](bool V) { PTO.MergeFunctions = V; });
  if (Key == "pto.InlinerThreshold")
    return Applier.applyI32([&](int32_t V) { PTO.InlinerThreshold = V; });
  if (Key == "pto.MaxDevirtIterations")
    return Applier.applyU32([&](uint32_t V) { PTO.MaxDevirtIterations = V; });
  if (Key == "pto.EagerlyInvalidateAnalyses")
    return Applier.applyBool(
        [&](bool V) { PTO.EagerlyInvalidateAnalyses = V; });
  if (Key == "pto.DevirtualizeSpeculatively")
    return Applier.applyBool(
        [&](bool V) { PTO.DevirtualizeSpeculatively = V; });

  return configError("unknown lto config key: " + Key);
}

Expected<Config> decodeConfigJSON(StringRef Text) {
  auto Value = json::parse(Text);
  if (!Value)
    return Value.takeError();
  return decodeVersionedJSON<Config>(
      *Value, kVersion, "lto config",
      [](Config &C, StringRef Key, const json::Value &Entry) {
        return applyEntry(C, Key, JSONEntryApplier{Entry, Key});
      });
}

} // namespace

std::string lto::serializeLTOConfig(const Config &Config) {
  json::Object Object{{"Version", kVersion}};
  JSONWriter Writer(Object);
  encodeConfigFields(Writer, Config);

  std::string Text;
  raw_string_ostream OS(Text);
  OS << json::Value(std::move(Object));
  return Text;
}

Expected<std::optional<Config>>
lto::readLTOConfigFromSummaryIndexIfPresent(const ModuleSummaryIndex &Index) {
  if (StringRef JSON = Index.getLTOConfigJSON(); !JSON.empty())
    return decodeConfigJSON(JSON);
  return std::nullopt;
}
