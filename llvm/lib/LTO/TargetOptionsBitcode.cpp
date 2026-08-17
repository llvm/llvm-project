//===- TargetOptionsBitcode.cpp - TargetOptions in bitcode ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes llvm::TargetOptions as module metadata that is stored in bitcode.
//
// Layout:
//   !llvm.lto.target_options = !{ !0 }
//   !0 = !{ i32 <version>, !1, !2, ... }
//   !1 = !{ !"<key>", <value> }
//
// Value kinds:
//   - i32 ConstantInt for bools, enums, and small integers
//   - MDString for std::string fields
//   - nested MDNode for structured fields such as MemoryBuffer
//
// Fields that cannot be represented in IR (such as callbacks) are
// intentionally omitted.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/TargetOptionsBitcode.h"

#include "BitcodeMetadataUtils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;
constexpr StringRef kEntryKind = "target options entry";

void encodeMemoryBuffer(MetadataWriter &Writer, StringRef Key,
                        const std::shared_ptr<MemoryBuffer> &Buffer) {
  if (!Buffer)
    return;

  Metadata *Fields[] = {
      getStringValue(Writer.getContext(), Buffer->getBufferIdentifier()),
      getStringValue(Writer.getContext(), Buffer->getBuffer())};
  Writer.putNode(Key, MDNode::get(Writer.getContext(), Fields));
}

Error decodeMemoryBuffer(std::shared_ptr<MemoryBuffer> &Buffer,
                         const MDNode &Entry) {
  Expected<MDNode *> Fields = getNodeField(Entry, kEntryKind);
  if (!Fields)
    return Fields.takeError();
  if ((*Fields)->getNumOperands() != 2)
    return metadataError(kEntryKind +
                         " memory buffer must contain an identifier and data");

  auto *Identifier = dyn_cast<MDString>((*Fields)->getOperand(0));
  auto *Data = dyn_cast<MDString>((*Fields)->getOperand(1));
  if (!Identifier || !Data)
    return metadataError(kEntryKind + " memory buffer fields must be strings");

  Buffer = MemoryBuffer::getMemBufferCopy(Data->getString(),
                                          Identifier->getString());
  return Error::success();
}

void encodeMCTargetOptions(MetadataWriter &Writer, const MCTargetOptions &MC) {
  // Keep this decomposition in sync with MCTargetOptions so that adding or
  // removing a field produces a compile error here.
  [[maybe_unused]] const auto &[MCRelaxAll, MCNoExecStack, MCFatalWarnings,
                                MCNoWarn, MCNoDeprecatedWarn, MCNoTypeCheck,
                                MCSaveTempLabels, MCIncrementalLinkerCompatible,
                                FDPIC, ShowMCEncoding, ShowMCInst, AsmVerbose,
                                PreserveAsmComments, Dwarf64, Crel,
                                ImplicitMapSyms, X86RelaxRelocations,
                                X86Sse2Avx, RelocSectionSym, OutputAsmVariant,
                                EmitDwarfUnwind, DwarfVersion,
                                MCUseDwarfDirectory, CompressDebugSections,
                                ABIName, AssemblyLanguage, SplitDwarfFile,
                                AsSecureLogFile, Argv0, CommandlineArgs,
                                IASSearchPaths, InstPrinterOptions,
                                EmitCompactUnwindNonCanonical, EmitSFrameUnwind,
                                PPCUseFullRegisterNames, LargeEHEncoding] = MC;

  Writer.putBool("mc.MCRelaxAll", MC.MCRelaxAll);
  Writer.putBool("mc.MCNoExecStack", MC.MCNoExecStack);
  Writer.putBool("mc.MCFatalWarnings", MC.MCFatalWarnings);
  Writer.putBool("mc.MCNoWarn", MC.MCNoWarn);
  Writer.putBool("mc.MCNoDeprecatedWarn", MC.MCNoDeprecatedWarn);
  Writer.putBool("mc.MCNoTypeCheck", MC.MCNoTypeCheck);
  Writer.putBool("mc.MCSaveTempLabels", MC.MCSaveTempLabels);
  Writer.putBool("mc.MCIncrementalLinkerCompatible",
                 MC.MCIncrementalLinkerCompatible);
  Writer.putBool("mc.FDPIC", MC.FDPIC);
  Writer.putBool("mc.ShowMCEncoding", MC.ShowMCEncoding);
  Writer.putBool("mc.ShowMCInst", MC.ShowMCInst);
  Writer.putBool("mc.AsmVerbose", MC.AsmVerbose);
  Writer.putBool("mc.PreserveAsmComments", MC.PreserveAsmComments);
  Writer.putBool("mc.Dwarf64", MC.Dwarf64);
  Writer.putBool("mc.Crel", MC.Crel);
  Writer.putBool("mc.ImplicitMapSyms", MC.ImplicitMapSyms);
  Writer.putBool("mc.X86RelaxRelocations", MC.X86RelaxRelocations);
  Writer.putBool("mc.X86Sse2Avx", MC.X86Sse2Avx);
  Writer.putI32("mc.RelocSectionSym", static_cast<int32_t>(MC.RelocSectionSym));
  if (MC.OutputAsmVariant)
    Writer.putI32("mc.OutputAsmVariant",
                  static_cast<int32_t>(*MC.OutputAsmVariant));
  Writer.putI32("mc.EmitDwarfUnwind", static_cast<int32_t>(MC.EmitDwarfUnwind));
  Writer.putI32("mc.DwarfVersion", MC.DwarfVersion);
  Writer.putI32("mc.MCUseDwarfDirectory",
                static_cast<int32_t>(MC.MCUseDwarfDirectory));
  Writer.putI32("mc.CompressDebugSections",
                static_cast<int32_t>(MC.CompressDebugSections));
  Writer.putString("mc.ABIName", MC.ABIName);
  Writer.putString("mc.AssemblyLanguage", MC.AssemblyLanguage);
  Writer.putString("mc.SplitDwarfFile", MC.SplitDwarfFile);
  Writer.putString("mc.AsSecureLogFile", MC.AsSecureLogFile);
  Writer.putString("mc.Argv0", MC.Argv0);
  Writer.putString("mc.CommandlineArgs", MC.CommandlineArgs);
  Writer.putStringList("mc.IASSearchPaths", MC.IASSearchPaths);
  Writer.putStringList("mc.InstPrinterOptions", MC.InstPrinterOptions);
  Writer.putBool("mc.EmitCompactUnwindNonCanonical",
                 MC.EmitCompactUnwindNonCanonical);
  Writer.putBool("mc.EmitSFrameUnwind", MC.EmitSFrameUnwind);
  Writer.putBool("mc.PPCUseFullRegisterNames", MC.PPCUseFullRegisterNames);
  Writer.putBool("mc.LargeEHEncoding", MC.LargeEHEncoding);
}

void encodeTargetOptionsFields(MetadataWriter &Writer,
                               const TargetOptions &Opt) {
  // Keep this decomposition in sync with TargetOptions. It intentionally
  // includes non-serializable fields so that adding or removing any field
  // produces a compile error here. New fields must either be serialized or
  // explicitly documented as non-serializable below.
  [[maybe_unused]] const auto
      &[BinutilsVersion, NoTrappingFPMath, EnableAIXExtendedAltivecABI,
        HonorSignDependentRoundingFPMathOption, NoZerosInBSS,
        GuaranteedTailCallOpt, StackSymbolOrdering, EnableFastISel,
        EnableGlobalISel, GlobalISelAbort, SwiftAsyncFramePointer, UseInitArray,
        DisableIntegratedAS, FunctionSections, DataSections,
        IgnoreXCOFFVisibility, XCOFFTracebackTable, UniqueSectionNames,
        UniqueBasicBlockSectionNames, SeparateNamedSections, TrapUnreachable,
        NoTrapAfterNoreturn, TLSSize, EmulatedTLS, EnableTLSDESC, EnableIPRA,
        EmitStackSizeSection, EnableMachineOutliner,
        EnableMachineFunctionSplitter, EnableStaticDataPartitioning,
        SupportsDefaultOutlining, EnableDefaultMachineVerifier, EmitAddrsig,
        BBAddrMap, BBSections, BBSectionsFuncListBuf,
        EmitCallGraphSection, EmitCallSiteInfo, SupportsDebugEntryValues,
        EnableDebugEntryValues, ValueTrackingVariableLocations,
        ForceDwarfFrameSection, XRayFunctionIndex, DebugStrictDwarf, Hotpatch,
        PPCGenScalarMASSEntries, JMCInstrument, EnableCFIFixup, MisExpect,
        XCOFFReadOnlyPointers, VerifyArgABICompliance, StackUsageFile,
        LoopAlignment, AllowFPOpFusion, ThreadModel, EABIVersion,
        DebuggerTuning, VecLib, ExceptionModel, MCOptions,
        ObjectFilenameForDebug] = Opt;

  Writer.putI32("BinutilsVersionMajor", Opt.BinutilsVersion.first);
  Writer.putI32("BinutilsVersionMinor", Opt.BinutilsVersion.second);

  Writer.putBool("NoTrappingFPMath", Opt.NoTrappingFPMath);
  Writer.putBool("EnableAIXExtendedAltivecABI",
                 Opt.EnableAIXExtendedAltivecABI);
  Writer.putBool("HonorSignDependentRoundingFPMathOption",
                 Opt.HonorSignDependentRoundingFPMathOption);
  Writer.putBool("NoZerosInBSS", Opt.NoZerosInBSS);
  Writer.putBool("GuaranteedTailCallOpt", Opt.GuaranteedTailCallOpt);
  Writer.putBool("StackSymbolOrdering", Opt.StackSymbolOrdering);
  Writer.putBool("EnableFastISel", Opt.EnableFastISel);
  Writer.putBool("EnableGlobalISel", Opt.EnableGlobalISel);
  Writer.putI32("GlobalISelAbort", static_cast<int32_t>(Opt.GlobalISelAbort));
  Writer.putI32("SwiftAsyncFramePointer",
                static_cast<int32_t>(Opt.SwiftAsyncFramePointer));
  Writer.putBool("UseInitArray", Opt.UseInitArray);
  Writer.putBool("DisableIntegratedAS", Opt.DisableIntegratedAS);
  Writer.putBool("FunctionSections", Opt.FunctionSections);
  Writer.putBool("DataSections", Opt.DataSections);
  Writer.putBool("IgnoreXCOFFVisibility", Opt.IgnoreXCOFFVisibility);
  Writer.putBool("XCOFFTracebackTable", Opt.XCOFFTracebackTable);
  Writer.putBool("UniqueSectionNames", Opt.UniqueSectionNames);
  Writer.putBool("UniqueBasicBlockSectionNames",
                 Opt.UniqueBasicBlockSectionNames);
  Writer.putBool("SeparateNamedSections", Opt.SeparateNamedSections);
  Writer.putBool("TrapUnreachable", Opt.TrapUnreachable);
  Writer.putBool("NoTrapAfterNoreturn", Opt.NoTrapAfterNoreturn);
  Writer.putI32("TLSSize", Opt.TLSSize);
  Writer.putBool("EmulatedTLS", Opt.EmulatedTLS);
  Writer.putBool("EnableTLSDESC", Opt.EnableTLSDESC);
  Writer.putBool("EnableIPRA", Opt.EnableIPRA);
  Writer.putBool("EmitStackSizeSection", Opt.EmitStackSizeSection);
  Writer.putBool("EnableMachineOutliner", Opt.EnableMachineOutliner);
  Writer.putBool("EnableMachineFunctionSplitter",
                 Opt.EnableMachineFunctionSplitter);
  Writer.putBool("EnableStaticDataPartitioning",
                 Opt.EnableStaticDataPartitioning);
  Writer.putBool("SupportsDefaultOutlining", Opt.SupportsDefaultOutlining);
  Writer.putBool("EnableDefaultMachineVerifier",
                 Opt.EnableDefaultMachineVerifier);
  Writer.putBool("EmitAddrsig", Opt.EmitAddrsig);
  Writer.putBool("BBAddrMap", Opt.BBAddrMap);
  Writer.putI32("BBSections", static_cast<int32_t>(Opt.BBSections));
  encodeMemoryBuffer(Writer, "BBSectionsFuncListBuf",
                     Opt.BBSectionsFuncListBuf);
  Writer.putBool("EmitCallGraphSection", Opt.EmitCallGraphSection);
  Writer.putBool("EmitCallSiteInfo", Opt.EmitCallSiteInfo);
  Writer.putBool("SupportsDebugEntryValues", Opt.SupportsDebugEntryValues);
  Writer.putBool("EnableDebugEntryValues", Opt.EnableDebugEntryValues);
  Writer.putBool("ValueTrackingVariableLocations",
                 Opt.ValueTrackingVariableLocations);
  Writer.putBool("ForceDwarfFrameSection", Opt.ForceDwarfFrameSection);
  Writer.putBool("XRayFunctionIndex", Opt.XRayFunctionIndex);
  Writer.putBool("DebugStrictDwarf", Opt.DebugStrictDwarf);
  Writer.putBool("Hotpatch", Opt.Hotpatch);
  Writer.putBool("PPCGenScalarMASSEntries", Opt.PPCGenScalarMASSEntries);
  Writer.putBool("JMCInstrument", Opt.JMCInstrument);
  Writer.putBool("EnableCFIFixup", Opt.EnableCFIFixup);
  Writer.putBool("MisExpect", Opt.MisExpect);
  Writer.putBool("XCOFFReadOnlyPointers", Opt.XCOFFReadOnlyPointers);
  Writer.putBool("VerifyArgABICompliance", Opt.VerifyArgABICompliance);

  Writer.putString("StackUsageFile", Opt.StackUsageFile);
  Writer.putI32("LoopAlignment", Opt.LoopAlignment);
  Writer.putI32("AllowFPOpFusion", static_cast<int32_t>(Opt.AllowFPOpFusion));
  Writer.putI32("ThreadModel", static_cast<int32_t>(Opt.ThreadModel));
  Writer.putI32("EABIVersion", static_cast<int32_t>(Opt.EABIVersion));
  Writer.putI32("DebuggerTuning", static_cast<int32_t>(Opt.DebuggerTuning));
  Writer.putI32("VecLib", static_cast<int32_t>(Opt.VecLib));
  Writer.putI32("ExceptionModel", static_cast<int32_t>(Opt.ExceptionModel));
  Writer.putString("ObjectFilenameForDebug", Opt.ObjectFilenameForDebug);

  encodeMCTargetOptions(Writer, Opt.MCOptions);
}

Error applyEntry(TargetOptions &Opt, StringRef Key, const MDNode &Entry) {
  EntryApplier Applier{Entry, kEntryKind};

  if (Key == "BinutilsVersionMajor")
    return Applier.applyI32([&](int32_t V) { Opt.BinutilsVersion.first = V; });
  if (Key == "BinutilsVersionMinor")
    return Applier.applyI32([&](int32_t V) { Opt.BinutilsVersion.second = V; });

  if (Key == "NoTrappingFPMath")
    return Applier.applyBool([&](bool V) { Opt.NoTrappingFPMath = V; });
  if (Key == "EnableAIXExtendedAltivecABI")
    return Applier.applyBool(
        [&](bool V) { Opt.EnableAIXExtendedAltivecABI = V; });
  if (Key == "HonorSignDependentRoundingFPMathOption")
    return Applier.applyBool(
        [&](bool V) { Opt.HonorSignDependentRoundingFPMathOption = V; });
  if (Key == "NoZerosInBSS")
    return Applier.applyBool([&](bool V) { Opt.NoZerosInBSS = V; });
  if (Key == "GuaranteedTailCallOpt")
    return Applier.applyBool([&](bool V) { Opt.GuaranteedTailCallOpt = V; });
  if (Key == "StackSymbolOrdering")
    return Applier.applyBool([&](bool V) { Opt.StackSymbolOrdering = V; });
  if (Key == "EnableFastISel")
    return Applier.applyBool([&](bool V) { Opt.EnableFastISel = V; });
  if (Key == "EnableGlobalISel")
    return Applier.applyBool([&](bool V) { Opt.EnableGlobalISel = V; });
  if (Key == "GlobalISelAbort")
    return Applier.applyI32([&](int32_t V) {
      Opt.GlobalISelAbort = static_cast<GlobalISelAbortMode>(V);
    });
  if (Key == "SwiftAsyncFramePointer")
    return Applier.applyI32([&](int32_t V) {
      Opt.SwiftAsyncFramePointer = static_cast<SwiftAsyncFramePointerMode>(V);
    });
  if (Key == "UseInitArray")
    return Applier.applyBool([&](bool V) { Opt.UseInitArray = V; });
  if (Key == "DisableIntegratedAS")
    return Applier.applyBool([&](bool V) { Opt.DisableIntegratedAS = V; });
  if (Key == "FunctionSections")
    return Applier.applyBool([&](bool V) { Opt.FunctionSections = V; });
  if (Key == "DataSections")
    return Applier.applyBool([&](bool V) { Opt.DataSections = V; });
  if (Key == "IgnoreXCOFFVisibility")
    return Applier.applyBool([&](bool V) { Opt.IgnoreXCOFFVisibility = V; });
  if (Key == "XCOFFTracebackTable")
    return Applier.applyBool([&](bool V) { Opt.XCOFFTracebackTable = V; });
  if (Key == "UniqueSectionNames")
    return Applier.applyBool([&](bool V) { Opt.UniqueSectionNames = V; });
  if (Key == "UniqueBasicBlockSectionNames")
    return Applier.applyBool(
        [&](bool V) { Opt.UniqueBasicBlockSectionNames = V; });
  if (Key == "SeparateNamedSections")
    return Applier.applyBool([&](bool V) { Opt.SeparateNamedSections = V; });
  if (Key == "TrapUnreachable")
    return Applier.applyBool([&](bool V) { Opt.TrapUnreachable = V; });
  if (Key == "NoTrapAfterNoreturn")
    return Applier.applyBool([&](bool V) { Opt.NoTrapAfterNoreturn = V; });
  if (Key == "TLSSize")
    return Applier.applyI32([&](int32_t V) { Opt.TLSSize = V; });
  if (Key == "EmulatedTLS")
    return Applier.applyBool([&](bool V) { Opt.EmulatedTLS = V; });
  if (Key == "EnableTLSDESC")
    return Applier.applyBool([&](bool V) { Opt.EnableTLSDESC = V; });
  if (Key == "EnableIPRA")
    return Applier.applyBool([&](bool V) { Opt.EnableIPRA = V; });
  if (Key == "EmitStackSizeSection")
    return Applier.applyBool([&](bool V) { Opt.EmitStackSizeSection = V; });
  if (Key == "EnableMachineOutliner")
    return Applier.applyBool([&](bool V) { Opt.EnableMachineOutliner = V; });
  if (Key == "EnableMachineFunctionSplitter")
    return Applier.applyBool(
        [&](bool V) { Opt.EnableMachineFunctionSplitter = V; });
  if (Key == "EnableStaticDataPartitioning")
    return Applier.applyBool(
        [&](bool V) { Opt.EnableStaticDataPartitioning = V; });
  if (Key == "SupportsDefaultOutlining")
    return Applier.applyBool([&](bool V) { Opt.SupportsDefaultOutlining = V; });
  if (Key == "EnableDefaultMachineVerifier")
    return Applier.applyBool(
        [&](bool V) { Opt.EnableDefaultMachineVerifier = V; });
  if (Key == "EmitAddrsig")
    return Applier.applyBool([&](bool V) { Opt.EmitAddrsig = V; });
  if (Key == "BBAddrMap")
    return Applier.applyBool([&](bool V) { Opt.BBAddrMap = V; });
  if (Key == "BBSections")
    return Applier.applyI32(
        [&](int32_t V) { Opt.BBSections = static_cast<BasicBlockSection>(V); });
  if (Key == "BBSectionsFuncListBuf")
    return decodeMemoryBuffer(Opt.BBSectionsFuncListBuf, Entry);
  if (Key == "EmitCallGraphSection")
    return Applier.applyBool([&](bool V) { Opt.EmitCallGraphSection = V; });
  if (Key == "EmitCallSiteInfo")
    return Applier.applyBool([&](bool V) { Opt.EmitCallSiteInfo = V; });
  if (Key == "SupportsDebugEntryValues")
    return Applier.applyBool([&](bool V) { Opt.SupportsDebugEntryValues = V; });
  if (Key == "EnableDebugEntryValues")
    return Applier.applyBool([&](bool V) { Opt.EnableDebugEntryValues = V; });
  if (Key == "ValueTrackingVariableLocations")
    return Applier.applyBool(
        [&](bool V) { Opt.ValueTrackingVariableLocations = V; });
  if (Key == "ForceDwarfFrameSection")
    return Applier.applyBool([&](bool V) { Opt.ForceDwarfFrameSection = V; });
  if (Key == "XRayFunctionIndex")
    return Applier.applyBool([&](bool V) { Opt.XRayFunctionIndex = V; });
  if (Key == "DebugStrictDwarf")
    return Applier.applyBool([&](bool V) { Opt.DebugStrictDwarf = V; });
  if (Key == "Hotpatch")
    return Applier.applyBool([&](bool V) { Opt.Hotpatch = V; });
  if (Key == "PPCGenScalarMASSEntries")
    return Applier.applyBool([&](bool V) { Opt.PPCGenScalarMASSEntries = V; });
  if (Key == "JMCInstrument")
    return Applier.applyBool([&](bool V) { Opt.JMCInstrument = V; });
  if (Key == "EnableCFIFixup")
    return Applier.applyBool([&](bool V) { Opt.EnableCFIFixup = V; });
  if (Key == "MisExpect")
    return Applier.applyBool([&](bool V) { Opt.MisExpect = V; });
  if (Key == "XCOFFReadOnlyPointers")
    return Applier.applyBool([&](bool V) { Opt.XCOFFReadOnlyPointers = V; });
  if (Key == "VerifyArgABICompliance")
    return Applier.applyBool([&](bool V) { Opt.VerifyArgABICompliance = V; });
  if (Key == "StackUsageFile")
    return Applier.applyString(
        [&](StringRef V) { Opt.StackUsageFile = V.str(); });
  if (Key == "LoopAlignment")
    return Applier.applyI32([&](int32_t V) { Opt.LoopAlignment = V; });
  if (Key == "AllowFPOpFusion")
    return Applier.applyI32([&](int32_t V) {
      Opt.AllowFPOpFusion = static_cast<FPOpFusion::FPOpFusionMode>(V);
    });
  if (Key == "ThreadModel")
    return Applier.applyI32([&](int32_t V) {
      Opt.ThreadModel = static_cast<ThreadModel::Model>(V);
    });
  if (Key == "EABIVersion")
    return Applier.applyI32(
        [&](int32_t V) { Opt.EABIVersion = static_cast<EABI>(V); });
  if (Key == "DebuggerTuning")
    return Applier.applyI32(
        [&](int32_t V) { Opt.DebuggerTuning = static_cast<DebuggerKind>(V); });
  if (Key == "VecLib")
    return Applier.applyI32(
        [&](int32_t V) { Opt.VecLib = static_cast<VectorLibrary>(V); });
  if (Key == "ExceptionModel")
    return Applier.applyI32([&](int32_t V) {
      Opt.ExceptionModel = static_cast<ExceptionHandling>(V);
    });
  if (Key == "ObjectFilenameForDebug")
    return Applier.applyString(
        [&](StringRef V) { Opt.ObjectFilenameForDebug = V.str(); });

  MCTargetOptions &MC = Opt.MCOptions;
  if (Key == "mc.MCRelaxAll")
    return Applier.applyBool([&](bool V) { MC.MCRelaxAll = V; });
  if (Key == "mc.MCNoExecStack")
    return Applier.applyBool([&](bool V) { MC.MCNoExecStack = V; });
  if (Key == "mc.MCFatalWarnings")
    return Applier.applyBool([&](bool V) { MC.MCFatalWarnings = V; });
  if (Key == "mc.MCNoWarn")
    return Applier.applyBool([&](bool V) { MC.MCNoWarn = V; });
  if (Key == "mc.MCNoDeprecatedWarn")
    return Applier.applyBool([&](bool V) { MC.MCNoDeprecatedWarn = V; });
  if (Key == "mc.MCNoTypeCheck")
    return Applier.applyBool([&](bool V) { MC.MCNoTypeCheck = V; });
  if (Key == "mc.MCSaveTempLabels")
    return Applier.applyBool([&](bool V) { MC.MCSaveTempLabels = V; });
  if (Key == "mc.MCIncrementalLinkerCompatible")
    return Applier.applyBool(
        [&](bool V) { MC.MCIncrementalLinkerCompatible = V; });
  if (Key == "mc.FDPIC")
    return Applier.applyBool([&](bool V) { MC.FDPIC = V; });
  if (Key == "mc.ShowMCEncoding")
    return Applier.applyBool([&](bool V) { MC.ShowMCEncoding = V; });
  if (Key == "mc.ShowMCInst")
    return Applier.applyBool([&](bool V) { MC.ShowMCInst = V; });
  if (Key == "mc.AsmVerbose")
    return Applier.applyBool([&](bool V) { MC.AsmVerbose = V; });
  if (Key == "mc.PreserveAsmComments")
    return Applier.applyBool([&](bool V) { MC.PreserveAsmComments = V; });
  if (Key == "mc.Dwarf64")
    return Applier.applyBool([&](bool V) { MC.Dwarf64 = V; });
  if (Key == "mc.Crel")
    return Applier.applyBool([&](bool V) { MC.Crel = V; });
  if (Key == "mc.ImplicitMapSyms")
    return Applier.applyBool([&](bool V) { MC.ImplicitMapSyms = V; });
  if (Key == "mc.X86RelaxRelocations")
    return Applier.applyBool([&](bool V) { MC.X86RelaxRelocations = V; });
  if (Key == "mc.X86Sse2Avx")
    return Applier.applyBool([&](bool V) { MC.X86Sse2Avx = V; });
  if (Key == "mc.RelocSectionSym")
    return Applier.applyI32([&](int32_t V) {
      MC.RelocSectionSym = static_cast<RelocSectionSymType>(V);
    });
  if (Key == "mc.OutputAsmVariant")
    return Applier.applyI32(
        [&](int32_t V) { MC.OutputAsmVariant = static_cast<unsigned>(V); });
  if (Key == "mc.EmitDwarfUnwind")
    return Applier.applyI32([&](int32_t V) {
      MC.EmitDwarfUnwind = static_cast<EmitDwarfUnwindType>(V);
    });
  if (Key == "mc.DwarfVersion")
    return Applier.applyI32([&](int32_t V) { MC.DwarfVersion = V; });
  if (Key == "mc.MCUseDwarfDirectory")
    return Applier.applyI32([&](int32_t V) {
      MC.MCUseDwarfDirectory = static_cast<MCTargetOptions::DwarfDirectory>(V);
    });
  if (Key == "mc.CompressDebugSections")
    return Applier.applyI32([&](int32_t V) {
      MC.CompressDebugSections = static_cast<DebugCompressionType>(V);
    });
  if (Key == "mc.ABIName")
    return Applier.applyString([&](StringRef V) { MC.ABIName = V.str(); });
  if (Key == "mc.AssemblyLanguage")
    return Applier.applyString(
        [&](StringRef V) { MC.AssemblyLanguage = V.str(); });
  if (Key == "mc.SplitDwarfFile")
    return Applier.applyString(
        [&](StringRef V) { MC.SplitDwarfFile = V.str(); });
  if (Key == "mc.AsSecureLogFile")
    return Applier.applyString(
        [&](StringRef V) { MC.AsSecureLogFile = V.str(); });
  if (Key == "mc.Argv0")
    return Applier.applyString([&](StringRef V) { MC.Argv0 = V.str(); });
  if (Key == "mc.CommandlineArgs")
    return Applier.applyString(
        [&](StringRef V) { MC.CommandlineArgs = V.str(); });
  if (Key == "mc.IASSearchPaths")
    return Applier.applyStringList(
        [&](std::vector<std::string> V) { MC.IASSearchPaths = std::move(V); });
  if (Key == "mc.InstPrinterOptions")
    return Applier.applyStringList([&](std::vector<std::string> V) {
      MC.InstPrinterOptions = std::move(V);
    });
  if (Key == "mc.EmitCompactUnwindNonCanonical")
    return Applier.applyBool(
        [&](bool V) { MC.EmitCompactUnwindNonCanonical = V; });
  if (Key == "mc.EmitSFrameUnwind")
    return Applier.applyBool([&](bool V) { MC.EmitSFrameUnwind = V; });
  if (Key == "mc.PPCUseFullRegisterNames")
    return Applier.applyBool([&](bool V) { MC.PPCUseFullRegisterNames = V; });
  if (Key == "mc.LargeEHEncoding")
    return Applier.applyBool([&](bool V) { MC.LargeEHEncoding = V; });

  return metadataError("unknown target options key: " + Key);
}

} // namespace

bool lto::hasEncodedTargetOptions(const Module &M) {
  return M.getNamedMetadata(TargetOptionsMetadataName) != nullptr;
}

Error lto::encodeTargetOptionsToModule(Module &M,
                                       const TargetOptions &Options) {
  MDNode *Root = encodeTargetOptionsAsNode(M.getContext(), Options);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(TargetOptionsMetadataName);
  NMD->clearOperands();
  NMD->addOperand(Root);
  return Error::success();
}

MDNode *lto::encodeTargetOptionsAsNode(LLVMContext &Ctx,
                                       const TargetOptions &Options) {
  SmallVector<Metadata *, 32> Entries;
  Entries.push_back(getI32Value(Ctx, kVersion));
  MetadataWriter Writer(Entries, Ctx);
  encodeTargetOptionsFields(Writer, Options);
  return MDNode::get(Ctx, Entries);
}

Expected<TargetOptions> lto::decodeTargetOptionsFromNode(const MDNode *Root) {
  return decodeVersionedMetadata<TargetOptions>(
      Root, kVersion, "target options",
      [](TargetOptions &Opt, StringRef Key, const MDNode &Entry) {
        return applyEntry(Opt, Key, Entry);
      });
}

Expected<TargetOptions> lto::decodeTargetOptionsFromModule(const Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata(TargetOptionsMetadataName);
  if (!NMD || NMD->getNumOperands() == 0)
    return metadataError("missing target options metadata");

  return decodeTargetOptionsFromNode(dyn_cast<MDNode>(NMD->getOperand(0)));
}
