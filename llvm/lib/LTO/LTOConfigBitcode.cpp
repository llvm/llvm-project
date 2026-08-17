//===- LTOConfigBitcode.cpp - lto::Config in bitcode ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes serializable lto::Config fields as module metadata in bitcode.
//
// Layout:
//   !llvm.lto.config = !{ !0 }
//   !0 = !{ i32 <version>, !1, !2, ... }
//   !1 = !{ !"<key>", <value> }
//
// Value kinds:
//   - i32 / i64 ConstantInt for scalars
//   - MDString for strings
//   - MDNode list of MDStrings for vector<string>
//   - nested MDNode for TargetOptions (via encodeTargetOptionsAsNode)
//
// Omitted fields (process-local / non-data):
//   LoadedPassPlugins, PreCodeGenPassesHook, DiagHandler, ResolutionFile,
//   PreOptModuleHook, PostPromoteModuleHook, PostInternalizeModuleHook,
//   PostImportModuleHook, PostOptModuleHook, PreCodeGenModuleHook,
//   CombinedIndexHook, GetSummaryIndexOutputStream, GetImportsListOutputArray,
//   GetCacheKeyOutputString
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"

#include "BitcodeMetadataUtils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/LTO/TargetOptionsBitcode.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;
constexpr StringRef kEntryKind = "lto config entry";

Error writeConfigBitcode(raw_ostream &Out, const Config &Config) {
  LLVMContext Ctx;
  Module M("llvm.lto.config", Ctx);
  if (Error Err = encodeLTOConfigToModule(M, Config))
    return Err;
  WriteBitcodeToFile(M, Out);
  return Error::success();
}

Expected<std::optional<Config>>
readConfigBitcodeIfPresent(MemoryBufferRef Buffer) {
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> M = parseBitcodeFile(Buffer, Ctx);
  if (!M)
    return M.takeError();
  if (!hasEncodedLTOConfig(**M))
    return std::nullopt;
  return decodeLTOConfigFromModule(**M);
}

Expected<Config> readConfigBitcode(MemoryBufferRef Buffer) {
  Expected<std::optional<Config>> Config = readConfigBitcodeIfPresent(Buffer);
  if (!Config)
    return Config.takeError();
  if (!*Config)
    return metadataError("missing lto config metadata");
  return std::move(**Config);
}

void encodePipelineTuningOptions(MetadataWriter &Writer,
                                 const PipelineTuningOptions &PTO) {
  Writer.putBool("pto.LoopInterleaving", PTO.LoopInterleaving);
  Writer.putBool("pto.LoopVectorization", PTO.LoopVectorization);
  Writer.putBool("pto.SLPVectorization", PTO.SLPVectorization);
  Writer.putBool("pto.LoopUnrolling", PTO.LoopUnrolling);
  Writer.putBool("pto.LoopInterchange", PTO.LoopInterchange);
  Writer.putBool("pto.LoopFusion", PTO.LoopFusion);
  Writer.putBool("pto.ForgetAllSCEVInLoopUnroll",
                 PTO.ForgetAllSCEVInLoopUnroll);
  Writer.putI32("pto.LicmMssaOptCap", PTO.LicmMssaOptCap);
  Writer.putI32("pto.LicmMssaNoAccForPromotionCap",
                PTO.LicmMssaNoAccForPromotionCap);
  Writer.putBool("pto.CallGraphProfile", PTO.CallGraphProfile);
  Writer.putBool("pto.UnifiedLTO", PTO.UnifiedLTO);
  Writer.putBool("pto.MergeFunctions", PTO.MergeFunctions);
  Writer.putI32("pto.InlinerThreshold", PTO.InlinerThreshold);
  Writer.putBool("pto.EagerlyInvalidateAnalyses",
                 PTO.EagerlyInvalidateAnalyses);
  Writer.putBool("pto.DevirtualizeSpeculatively",
                 PTO.DevirtualizeSpeculatively);
}

void encodeRemarksHotnessThreshold(MetadataWriter &Writer,
                                   const std::optional<uint64_t> &Threshold) {
  int32_t Mode = 0;
  uint64_t Value = 0;
  if (!Threshold.has_value()) {
    Mode = 2; // auto
  } else if (*Threshold == 0) {
    Mode = 0; // disabled
  } else {
    Mode = 1; // manual
    Value = *Threshold;
  }
  LLVMContext &Ctx = Writer.getContext();
  Metadata *Ops[] = {getI32Value(Ctx, Mode), getI64Value(Ctx, Value)};
  Writer.putEntry("RemarksHotnessThreshold", MDNode::get(Ctx, Ops));
}

void encodeOptionalCodeModel(MetadataWriter &Writer,
                             const std::optional<CodeModel::Model> &CodeModel) {
  if (!CodeModel)
    return;
  Writer.putI32("CodeModel", static_cast<int32_t>(*CodeModel));
}

void encodeConfigFields(MetadataWriter &Writer, const Config &C) {
  // Keep this decomposition in sync with Config. It intentionally includes
  // non-serializable fields so that adding or removing any field produces a
  // compile error here. New fields must either be added to encodeConfigFields
  // and applyEntry or explicitly documented as non-serializable below.
  [[maybe_unused]] const auto
      &[CPU, Options, MAttrs, MllvmArgs,
        LoadedPassPlugins, // Non-serializable process-local pointers.
        PassPluginFilenames,
        PreCodeGenPassesHook, // Non-serializable callback.
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
        GetSummaryIndexOutputStream, GetImportsListOutputArray,
        GetCacheKeyOutputString] = C;

  Writer.putString("CPU", C.CPU);
  Writer.putNode("Options",
                 encodeTargetOptionsAsNode(Writer.getContext(), C.Options));
  Writer.putStringList("MAttrs", C.MAttrs);
  Writer.putStringList("MllvmArgs", C.MllvmArgs);
  Writer.putStringList("PassPluginFilenames", C.PassPluginFilenames);

  Writer.putBool("RelocModel.HasValue", C.RelocModel.has_value());
  if (C.RelocModel)
    Writer.putI32("RelocModel", static_cast<int32_t>(*C.RelocModel));
  encodeOptionalCodeModel(Writer, C.CodeModel);

  Writer.putI32("CGOptLevel", static_cast<int32_t>(C.CGOptLevel));
  Writer.putI32("CGFileType", static_cast<int32_t>(C.CGFileType));
  Writer.putI32("OptLevel", C.OptLevel);

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
  encodeRemarksHotnessThreshold(Writer, C.RemarksHotnessThreshold);
  Writer.putString("RemarksFormat", C.RemarksFormat);
  Writer.putBool("DebugPassManager", C.DebugPassManager);
  Writer.putString("StatsFile", C.StatsFile);
  Writer.putStringList("ThinLTOModulesToCompile", C.ThinLTOModulesToCompile);
  Writer.putBool("TimeTraceEnabled", C.TimeTraceEnabled);
  Writer.putI32("TimeTraceGranularity", C.TimeTraceGranularity);
  Writer.putBool("ShouldDiscardValueNames", C.ShouldDiscardValueNames);
  Writer.putBool("AddFSDiscriminator", C.AddFSDiscriminator);

  encodePipelineTuningOptions(Writer, C.PTO);
}

Error applyEntry(Config &C, StringRef Key, const MDNode &Entry) {
  EntryApplier Applier{Entry, kEntryKind};

  if (Key == "CPU")
    return Applier.applyString([&](StringRef V) { C.CPU = V.str(); });
  if (Key == "Options") {
    auto Node = getNodeField(Entry, kEntryKind);
    if (!Node)
      return Node.takeError();
    auto Opt = decodeTargetOptionsFromNode(*Node);
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
    return Applier.applyI32(
        [&](int32_t V) { C.RelocModel = static_cast<Reloc::Model>(V); });
  if (Key == "RelocModel.HasValue")
    return Applier.applyBool([&](bool V) {
      if (!V)
        C.RelocModel = std::nullopt;
    });
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
    return Applier.applyI32([&](int32_t V) { C.OptLevel = V; });

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
  if (Key == "RemarksHotnessThreshold") {
    auto Node = getNodeField(Entry, kEntryKind);
    if (!Node)
      return Node.takeError();
    if ((*Node)->getNumOperands() != 2)
      return metadataError("RemarksHotnessThreshold must have mode and value");
    auto *Mode = mdconst::dyn_extract<ConstantInt>((*Node)->getOperand(0));
    auto *Value = mdconst::dyn_extract<ConstantInt>((*Node)->getOperand(1));
    if (!Mode || !Mode->getType()->isIntegerTy(32) || !Value ||
        !Value->getType()->isIntegerTy(64))
      return metadataError("malformed RemarksHotnessThreshold metadata");
    switch (Mode->getZExtValue()) {
    case 0:
      C.RemarksHotnessThreshold = 0;
      break;
    case 1:
      C.RemarksHotnessThreshold = Value->getZExtValue();
      break;
    case 2:
      C.RemarksHotnessThreshold = std::nullopt;
      break;
    default:
      return metadataError("invalid RemarksHotnessThreshold mode");
    }
    return Error::success();
  }
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
    return Applier.applyI32([&](int32_t V) { C.TimeTraceGranularity = V; });
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
    return Applier.applyI32([&](int32_t V) { PTO.LicmMssaOptCap = V; });
  if (Key == "pto.LicmMssaNoAccForPromotionCap")
    return Applier.applyI32(
        [&](int32_t V) { PTO.LicmMssaNoAccForPromotionCap = V; });
  if (Key == "pto.CallGraphProfile")
    return Applier.applyBool([&](bool V) { PTO.CallGraphProfile = V; });
  if (Key == "pto.UnifiedLTO")
    return Applier.applyBool([&](bool V) { PTO.UnifiedLTO = V; });
  if (Key == "pto.MergeFunctions")
    return Applier.applyBool([&](bool V) { PTO.MergeFunctions = V; });
  if (Key == "pto.InlinerThreshold")
    return Applier.applyI32([&](int32_t V) { PTO.InlinerThreshold = V; });
  if (Key == "pto.EagerlyInvalidateAnalyses")
    return Applier.applyBool(
        [&](bool V) { PTO.EagerlyInvalidateAnalyses = V; });
  if (Key == "pto.DevirtualizeSpeculatively")
    return Applier.applyBool(
        [&](bool V) { PTO.DevirtualizeSpeculatively = V; });

  return metadataError("unknown lto config key: " + Key);
}

Expected<Config> decodeConfigFromRoot(const MDNode *Root) {
  return decodeVersionedMetadata<Config>(
      Root, kVersion, "lto config",
      [](Config &C, StringRef Key, const MDNode &Entry) {
        return applyEntry(C, Key, Entry);
      });
}

} // namespace

bool lto::hasEncodedLTOConfig(const Module &M) {
  return M.getNamedMetadata(LTOConfigMetadataName) != nullptr;
}

Error lto::encodeLTOConfigToModule(Module &M, const Config &Config) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<Metadata *, 64> Entries;
  Entries.push_back(getI32Value(Ctx, kVersion));
  MetadataWriter Writer(Entries, Ctx);
  encodeConfigFields(Writer, Config);

  MDNode *Root = MDNode::get(Ctx, Entries);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(LTOConfigMetadataName);
  NMD->clearOperands();
  NMD->addOperand(Root);
  return Error::success();
}

Expected<Config> lto::decodeLTOConfigFromModule(const Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata(LTOConfigMetadataName);
  if (!NMD || NMD->getNumOperands() == 0)
    return metadataError("missing lto config metadata");
  return decodeConfigFromRoot(dyn_cast<MDNode>(NMD->getOperand(0)));
}

Error lto::writeLTOConfigToFile(StringRef Path, const Config &Config) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, "cannot open LTO config file '%s'",
                             Path.str().c_str());
  if (Error Err = writeConfigBitcode(OS, Config))
    return Err;
  OS.close();
  if (OS.has_error())
    return createStringError(OS.error(), "cannot write LTO config file '%s'",
                             Path.str().c_str());
  return Error::success();
}

Expected<Config> lto::readLTOConfigFromFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(),
                             "cannot read LTO config file '%s'",
                             Path.str().c_str());

  return readConfigBitcode((*Buffer)->getMemBufferRef());
}

Error lto::writeIndexWithLTOConfigToFile(
    const ModuleSummaryIndex &Index, const Config &Config, raw_ostream &Out,
    const ModuleToSummariesForIndexTy *ModuleToSummariesForIndex,
    const GVSummaryPtrSet *DecSummaries) {
  LLVMContext Ctx;
  Module MetadataModule("llvm.lto.config", Ctx);
  if (Error Err = encodeLTOConfigToModule(MetadataModule, Config))
    return Err;
  writeIndexToFile(Index, Out, ModuleToSummariesForIndex, DecSummaries,
                   &MetadataModule);
  return Error::success();
}

Expected<Config> lto::readLTOConfigFromSummaryIndex(MemoryBufferRef Buffer) {
  return readConfigBitcode(Buffer);
}

Expected<std::optional<Config>>
lto::readLTOConfigFromSummaryIndexIfPresent(MemoryBufferRef Buffer) {
  return readConfigBitcodeIfPresent(Buffer);
}
