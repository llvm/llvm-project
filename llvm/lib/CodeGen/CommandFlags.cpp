//===-- CommandFlags.cpp - Command Line Flags Interface ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains codegen-specific flags that are shared between different
// command line tools. The tools "llc" and "opt" both use this file to prevent
// flag duplication.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <memory>
#include <optional>
#include <system_error>

using namespace llvm;

#define OPTIONS_STRUCT_DECL
#include "CodeGenToolOptions.inc"

#define OPTIONS_STRUCT_DEFS
#include "CodeGenToolOptions.inc"

static const CodeGenToolOptions &getCodeGenToolOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<CodeGenToolOptions>();
}

#define CGOPT(TY, NAME)                                                        \
  static cl::opt<TY> *NAME##View;                                              \
  TY codegen::get##NAME(const cl::LibraryOptionsContext &) {                   \
    assert(NAME##View && "Flag not registered.");                              \
    return *NAME##View;                                                        \
  }

#define CGLIST(TY, NAME)                                                       \
  static cl::list<TY> *NAME##View;                                             \
  std::vector<TY> codegen::get##NAME(const cl::LibraryOptionsContext &) {      \
    assert(NAME##View && "Flag not registered.");                              \
    return *NAME##View;                                                        \
  }

// Temporary macro for incremental transition to std::optional.
#define CGOPT_EXP(TY, NAME)                                                    \
  CGOPT(TY, NAME)                                                              \
  std::optional<TY> codegen::getExplicit##NAME(                                \
      const cl::LibraryOptionsContext &) {                                     \
    if (NAME##View->getNumOccurrences()) {                                     \
      TY res = *NAME##View;                                                    \
      return res;                                                              \
    }                                                                          \
    return std::nullopt;                                                       \
  }

CGOPT(std::string, MArch)
CGOPT(std::string, MCPU)
CGOPT(std::string, MTune)
CGLIST(std::string, MAttrs)
CGOPT_EXP(Reloc::Model, RelocModel)
CGOPT_EXP(CodeModel::Model, CodeModel)
CGOPT_EXP(uint64_t, LargeDataThreshold)
CGOPT_EXP(CodeGenFileType, FileType)
CGOPT(FloatABI::ABIType, FloatABIForCalls)
CGOPT_EXP(bool, DataSections)
CGOPT_EXP(bool, FunctionSections)
CGOPT_EXP(bool, EmulatedTLS)
CGOPT(codegen::SaveStatsMode, SaveStats)

#define CGBINDOPT(NAME)                                                        \
  do {                                                                         \
    NAME##View = std::addressof(NAME);                                         \
  } while (0)

codegen::RegisterCodeGenFlags::RegisterCodeGenFlags() {
  static cl::opt<std::string> MArch(
      "march", cl::desc("Architecture to generate code for (see --version)"));
  CGBINDOPT(MArch);

  static cl::opt<std::string> MCPU(
      "mcpu", cl::desc("Target a specific cpu type (-mcpu=help for details)"),
      cl::value_desc("cpu-name"), cl::init(""));
  CGBINDOPT(MCPU);

  static cl::list<std::string> MAttrs(
      "mattr", cl::CommaSeparated,
      cl::desc("Target specific attributes (-mattr=help for details)"),
      cl::value_desc("a1,+a2,-a3,..."));
  CGBINDOPT(MAttrs);

  static cl::opt<Reloc::Model> RelocModel(
      "relocation-model", cl::desc("Choose relocation model"),
      cl::values(
          clEnumValN(Reloc::Static, "static", "Non-relocatable code"),
          clEnumValN(Reloc::PIC_, "pic",
                     "Fully relocatable, position independent code"),
          clEnumValN(Reloc::DynamicNoPIC, "dynamic-no-pic",
                     "Relocatable external references, non-relocatable code"),
          clEnumValN(
              Reloc::ROPI, "ropi",
              "Code and read-only data relocatable, accessed PC-relative"),
          clEnumValN(
              Reloc::RWPI, "rwpi",
              "Read-write data relocatable, accessed relative to static base"),
          clEnumValN(Reloc::ROPI_RWPI, "ropi-rwpi",
                     "Combination of ropi and rwpi")));
  CGBINDOPT(RelocModel);

  static cl::opt<CodeModel::Model> CodeModel(
      "code-model", cl::desc("Choose code model"),
      cl::values(clEnumValN(CodeModel::Tiny, "tiny", "Tiny code model"),
                 clEnumValN(CodeModel::Small, "small", "Small code model"),
                 clEnumValN(CodeModel::Kernel, "kernel", "Kernel code model"),
                 clEnumValN(CodeModel::Medium, "medium", "Medium code model"),
                 clEnumValN(CodeModel::Large, "large", "Large code model")));
  CGBINDOPT(CodeModel);

  static cl::opt<uint64_t> LargeDataThreshold(
      "large-data-threshold",
      cl::desc("Choose large data threshold for x86_64 medium code model"),
      cl::init(0));
  CGBINDOPT(LargeDataThreshold);

  static cl::opt<CodeGenFileType> FileType(
      "filetype", cl::init(CodeGenFileType::AssemblyFile),
      cl::desc(
          "Choose a file type (not all types are supported by all targets):"),
      cl::values(clEnumValN(CodeGenFileType::AssemblyFile, "asm",
                            "Emit an assembly ('.s') file"),
                 clEnumValN(CodeGenFileType::ObjectFile, "obj",
                            "Emit a native object ('.o') file"),
                 clEnumValN(CodeGenFileType::Null, "null",
                            "Emit nothing, for performance testing")));
  CGBINDOPT(FileType);

  static cl::opt<FloatABI::ABIType> FloatABIForCalls(
      "float-abi", cl::desc("Choose float ABI type"),
      cl::init(FloatABI::Default),
      cl::values(clEnumValN(FloatABI::Default, "default",
                            "Target default float ABI type"),
                 clEnumValN(FloatABI::Soft, "soft",
                            "Soft float ABI (implied by -soft-float)"),
                 clEnumValN(FloatABI::Hard, "hard",
                            "Hard float ABI (uses FP registers)")));
  CGBINDOPT(FloatABIForCalls);

  static cl::opt<bool> DataSections(
      "data-sections", cl::desc("Emit data into separate sections"),
      cl::init(false));
  CGBINDOPT(DataSections);

  static cl::opt<bool> FunctionSections(
      "function-sections", cl::desc("Emit functions into separate sections"),
      cl::init(false));
  CGBINDOPT(FunctionSections);

  static cl::opt<bool> EmulatedTLS(
      "emulated-tls", cl::desc("Use emulated TLS model"), cl::init(false));
  CGBINDOPT(EmulatedTLS);

}

ArrayRef<const cl::LibraryOptions *> codegen::getToolOptions() {
  static constexpr opt::LibraryOptionsParser Parser(
      std::in_place_type<CodeGenToolOptions>);
  assert(mc::getToolOptions().size() == 1);
  static const cl::LibraryOptions *const Libraries[] = {mc::getToolOptions()[0],
                                                        &Parser};
  return Libraries;
}

codegen::RegisterMTuneFlag::RegisterMTuneFlag() {
  static cl::opt<std::string> MTune(
      "mtune",
      cl::desc("Tune for a specific CPU microarchitecture (-mtune=help for "
               "details)"),
      cl::value_desc("tune-cpu-name"), cl::init(""));
  CGBINDOPT(MTune);
}

codegen::RegisterSaveStatsFlag::RegisterSaveStatsFlag() {
  static cl::opt<SaveStatsMode> SaveStats(
      "save-stats",
      cl::desc(
          "Save LLVM statistics to a file in the current directory"
          "(`-save-stats`/`-save-stats=cwd`) or the directory of the output"
          "file (`-save-stats=obj`). (default: cwd)"),
      cl::values(clEnumValN(SaveStatsMode::Cwd, "cwd",
                            "Save to the current working directory"),
                 clEnumValN(SaveStatsMode::Cwd, "", ""),
                 clEnumValN(SaveStatsMode::Obj, "obj",
                            "Save to the output file directory")),
      cl::init(SaveStatsMode::None), cl::ValueOptional);
  CGBINDOPT(SaveStats);
}

static BasicBlockSection getBBSectionsMode(const CodeGenToolOptions &Opts,
                                           TargetOptions &Options) {
  if (Opts.basic_block_sections == "all")
    return BasicBlockSection::All;
  else if (Opts.basic_block_sections == "none")
    return BasicBlockSection::None;
  else {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
        MemoryBuffer::getFile(Opts.basic_block_sections);
    if (!MBOrErr) {
      errs() << "Error loading basic block sections function list file: "
             << MBOrErr.getError().message() << "\n";
    } else {
      Options.BBSectionsFuncListBuf = std::move(*MBOrErr);
    }
    return BasicBlockSection::List;
  }
}

// Common utility function tightly tied to the options listed here. Initializes
// a TargetOptions object with CodeGen options and returns it.
TargetOptions
codegen::getTargetOptions(const cl::LibraryOptionsContext &LibraryOptions,
                          const Triple &TheTriple) {
  const CodeGenToolOptions &Opts = LibraryOptions.get<CodeGenToolOptions>();
  TargetOptions Options;
  Options.NoZerosInBSS = Opts.nozero_initialized_in_bss;
  Options.GuaranteedTailCallOpt = Opts.tailcallopt;
  Options.StackSymbolOrdering = Opts.stack_symbol_ordering;
  Options.UseInitArray = !Opts.use_ctors;
  Options.DataSections = getExplicitDataSections(LibraryOptions)
                             .value_or(TheTriple.hasDefaultDataSections());
  Options.FunctionSections = getFunctionSections(LibraryOptions);
  Options.IgnoreXCOFFVisibility = Opts.ignore_xcoff_visibility;
  Options.XCOFFTracebackTable = Opts.xcoff_traceback_table;
  Options.BBAddrMap = Opts.basic_block_address_map;
  Options.BBSections = getBBSectionsMode(Opts, Options);
  Options.UniqueSectionNames = Opts.unique_section_names;
  Options.UniqueBasicBlockSectionNames = Opts.unique_basic_block_section_names;
  Options.SeparateNamedSections = Opts.separate_named_sections;
  Options.TLSSize = Opts.tls_size;
  Options.EmulatedTLS = getExplicitEmulatedTLS(LibraryOptions)
                            .value_or(TheTriple.hasDefaultEmulatedTLS());
  Options.EnableTLSDESC =
      valueOr(Opts.enable_tlsdesc, TheTriple.hasDefaultTLSDESC());
  Options.VecLib = Opts.vector_library;
  Options.EmitStackSizeSection = Opts.stack_size_section;
  Options.EnableStaticDataPartitioning = Opts.partition_static_data_sections;
  Options.EmitAddrsig = Opts.addrsig;
  Options.EmitCallGraphSection = Opts.call_graph_section;
  Options.EmitCallSiteInfo = Opts.emit_call_site_info;
  Options.EnableDebugEntryValues = Opts.debug_entry_values;
  Options.ForceDwarfFrameSection = Opts.force_dwarf_frame_section;
  Options.XRayFunctionIndex = Opts.xray_function_index;
  Options.DebugStrictDwarf = Opts.strict_dwarf;
  Options.LoopAlignment = Opts.align_loops;
  Options.JMCInstrument = Opts.enable_jmc_instrument;
  Options.XCOFFReadOnlyPointers = Opts.mxcoff_roptr;

  Options.MCOptions = mc::getMCTargetOptions(LibraryOptions);

  Options.SwiftAsyncFramePointer = Opts.swift_async_fp;
  return Options;
}

std::string codegen::getCPUStr(const cl::LibraryOptionsContext &C) {
  std::string MCPU = getMCPU(C);

  // If user asked for the 'native' CPU, autodetect here. If auto-detection
  // fails, this will set the CPU to an empty string which tells the target to
  // pick a basic default.
  if (MCPU == "native")
    return std::string(sys::getHostCPUName());

  return MCPU;
}

std::string codegen::getTuneCPUStr(const cl::LibraryOptionsContext &C) {
  std::string TuneCPU = getMTune(C);

  // If user asked for the 'native' tune CPU, autodetect here. If auto-detection
  // fails, this will set the tune CPU to an empty string which tells the target
  // to pick a basic default.
  if (TuneCPU == "native")
    return std::string(sys::getHostCPUName());

  return TuneCPU;
}

std::string codegen::getFeaturesStr(const cl::LibraryOptionsContext &C) {
  SubtargetFeatures Features;

  // If user asked for the 'native' CPU, we need to autodetect features.
  // This is necessary for x86 where the CPU might not support all the
  // features the autodetected CPU name lists in the target. For example,
  // not all Sandybridge processors support AVX.
  if (getMCPU(C) == "native")
    for (const auto &[Feature, IsEnabled] : sys::getHostCPUFeatures())
      Features.AddFeature(Feature, IsEnabled);

  for (auto const &MAttr : getMAttrs(C))
    Features.AddFeature(MAttr);

  return Features.getString();
}

std::vector<std::string>
codegen::getFeatureList(const cl::LibraryOptionsContext &C) {
  SubtargetFeatures Features;

  // If user asked for the 'native' CPU, we need to autodetect features.
  // This is necessary for x86 where the CPU might not support all the
  // features the autodetected CPU name lists in the target. For example,
  // not all Sandybridge processors support AVX.
  if (getMCPU(C) == "native")
    for (const auto &[Feature, IsEnabled] : sys::getHostCPUFeatures())
      Features.AddFeature(Feature, IsEnabled);

  for (auto const &MAttr : getMAttrs(C))
    Features.AddFeature(MAttr);

  return Features.getFeatures();
}

void codegen::renderBoolStringAttr(AttrBuilder &B, StringRef Name, bool Val) {
  B.addAttribute(Name, Val ? "true" : "false");
}

void codegen::setFunctionAttributes(Function &F, StringRef CPU,
                                    StringRef Features, StringRef TuneCPU) {
  auto &Ctx = F.getContext();
  const CodeGenToolOptions &Opts = getCodeGenToolOptions(Ctx);
  AttributeList Attrs = F.getAttributes();
  AttrBuilder NewAttrs(Ctx);

  if (!CPU.empty() && !F.hasFnAttribute("target-cpu"))
    NewAttrs.addAttribute("target-cpu", CPU);
  if (!TuneCPU.empty() && !F.hasFnAttribute("tune-cpu"))
    NewAttrs.addAttribute("tune-cpu", TuneCPU);
  if (!Features.empty()) {
    // Append the command line features to any that are already on the function.
    StringRef OldFeatures =
        F.getFnAttribute("target-features").getValueAsString();
    if (OldFeatures.empty())
      NewAttrs.addAttribute("target-features", Features);
    else {
      SmallString<256> Appended(OldFeatures);
      Appended.push_back(',');
      Appended.append(Features);
      NewAttrs.addAttribute("target-features", Appended);
    }
  }
  if (Opts.frame_pointer && !F.hasFnAttribute("frame-pointer")) {
    if (*Opts.frame_pointer == FramePointerKind::All)
      NewAttrs.addAttribute("frame-pointer", "all");
    else if (*Opts.frame_pointer == FramePointerKind::NonLeaf)
      NewAttrs.addAttribute("frame-pointer", "non-leaf");
    else if (*Opts.frame_pointer == FramePointerKind::NonLeafNoReserve)
      NewAttrs.addAttribute("frame-pointer", "non-leaf-no-reserve");
    else if (*Opts.frame_pointer == FramePointerKind::Reserved)
      NewAttrs.addAttribute("frame-pointer", "reserved");
    else if (*Opts.frame_pointer == FramePointerKind::None)
      NewAttrs.addAttribute("frame-pointer", "none");
  }
  if (Opts.disable_tail_calls != BoolOrDefault::Default)
    NewAttrs.addAttribute("disable-tail-calls",
                          toStringRef(valueOr(Opts.disable_tail_calls, false)));
  if (Opts.stackrealign)
    NewAttrs.addAttribute("stackrealign");

  if ((Opts.denormal_fp_math || Opts.denormal_fp_math_f32) &&
      !F.hasFnAttribute(Attribute::DenormalFPEnv)) {
    DenormalMode::DenormalModeKind DenormKind =
        Opts.denormal_fp_math.value_or(DenormalMode::IEEE);
    DenormalMode::DenormalModeKind DenormKindF32 =
        Opts.denormal_fp_math_f32.value_or(DenormalMode::Invalid);

    DenormalFPEnv FPEnv(DenormalMode{DenormKind, DenormKind},
                        DenormalMode{DenormKindF32, DenormKindF32});
    // FIXME: Command line flag should expose separate input/output modes.
    NewAttrs.addDenormalFPEnvAttr(FPEnv);
  }

  if (Opts.trap_func)
    for (auto &B : F)
      for (auto &I : B)
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (const auto *F = Call->getCalledFunction())
            if (F->getIntrinsicID() == Intrinsic::debugtrap ||
                F->getIntrinsicID() == Intrinsic::trap)
              Call->addFnAttr(
                  Attribute::get(Ctx, "trap-func-name", *Opts.trap_func));

  // Let NewAttrs override Attrs.
  F.setAttributes(Attrs.addFnAttributes(Ctx, NewAttrs));
}

void codegen::setFunctionAttributes(Module &M, StringRef CPU,
                                    StringRef Features, StringRef TuneCPU) {
  const cl::LibraryOptionsContext &C = M.getContext().getLibraryOptions();
  // Synthesize the "float-abi" module flag from the -float-abi option.
  FloatABI::ABIType ABI = getFloatABIForCalls(C);
  if (ABI != FloatABI::Default) {
    if (auto *Existing =
            dyn_cast_or_null<MDString>(M.getModuleFlag("float-abi"))) {
      // The module already records a float ABI; -float-abi must not contradict
      // it.
      if (Existing->getString() != FloatABI::getABITypeName(ABI))
        reportFatalUsageError(
            "-float-abi=" + FloatABI::getABITypeName(ABI) +
            " conflicts with the \"float-abi\" module flag \"" +
            Existing->getString() + "\"");
    } else {
      M.addModuleFlag(
          Module::Error, "float-abi",
          MDString::get(M.getContext(), FloatABI::getABITypeName(ABI)));
    }
  }

  // Synthesize the "exception-model" module flag from the -exception-model
  // option.
  ExceptionHandling EH = getCodeGenToolOptions(M.getContext()).exception_model;
  if (EH != ExceptionHandling::Default) {
    if (auto *Existing =
            dyn_cast_or_null<MDString>(M.getModuleFlag("exception-model"))) {
      // The module already records an exception model; -exception-model must
      // not contradict it.
      if (Existing->getString() != getExceptionModelName(EH)) {
        reportFatalUsageError(
            "-exception-model=" + getExceptionModelName(EH) +
            " conflicts with the \"exception-model\" module flag \"" +
            Existing->getString() + "\"");
      }
    } else {
      M.addModuleFlag(Module::Error, "exception-model",
                      MDString::get(M.getContext(), getExceptionModelName(EH)));
    }
  }

  // Synthesize the "target-abi" module flag from the -target-abi option.
  //
  // FIXME: verifyOptionsConsistency validates consistency for target-abi. We
  // should consistently handle all ABI module flags either here or there.
  StringRef ABIName = mc::getABIName(C);
  if (!ABIName.empty() && !M.getModuleFlag("target-abi")) {
    M.addModuleFlag(Module::Error, "target-abi",
                    MDString::get(M.getContext(), ABIName));
  }

  for (Function &F : M)
    setFunctionAttributes(F, CPU, Features, TuneCPU);
}

Expected<std::unique_ptr<TargetMachine>> codegen::createTargetMachineForTriple(
    const cl::LibraryOptionsContext &LibraryOptions, const Triple &TargetTriple,
    CodeGenOptLevel OptLevel) {
  // lookupTarget may mutate the triple, so we need a copy.
  Triple TheTriple(TargetTriple);
  std::string Error;
  const auto *TheTarget = TargetRegistry::lookupTarget(
      codegen::getMArch(LibraryOptions), TheTriple, Error);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(), Error);
  TargetOptions Options = codegen::getTargetOptions(LibraryOptions, TheTriple);
  Options.MCOptions.LibraryOptions = &LibraryOptions;
  auto *Target = TheTarget->createTargetMachine(
      TheTriple, codegen::getCPUStr(LibraryOptions),
      codegen::getFeaturesStr(LibraryOptions), Options,
      codegen::getExplicitRelocModel(LibraryOptions),
      codegen::getExplicitCodeModel(LibraryOptions), OptLevel);
  if (!Target)
    return createStringError(inconvertibleErrorCode(),
                             Twine("could not allocate target machine for ") +
                                 TheTriple.str());
  return std::unique_ptr<TargetMachine>(Target);
}

void codegen::maybeEnableStatistics(const cl::LibraryOptionsContext &C) {
  if (getSaveStats(C) == SaveStatsMode::None)
    return;

  llvm::EnableStatistics(false);
}

int codegen::maybeSaveStatistics(const cl::LibraryOptionsContext &C,
                                 StringRef OutputFilename, StringRef ToolName) {
  auto SaveStatsValue = getSaveStats(C);
  if (SaveStatsValue == codegen::SaveStatsMode::None)
    return 0;

  SmallString<128> StatsFilename;
  if (SaveStatsValue == codegen::SaveStatsMode::Obj) {
    StatsFilename = OutputFilename;
    llvm::sys::path::remove_filename(StatsFilename);
  } else {
    assert(SaveStatsValue == codegen::SaveStatsMode::Cwd &&
           "Should have been a valid --save-stats value");
  }

  auto BaseName = llvm::sys::path::filename(OutputFilename);
  llvm::sys::path::append(StatsFilename, BaseName);
  llvm::sys::path::replace_extension(StatsFilename, "stats");

  auto FileFlags = llvm::sys::fs::OF_TextWithCRLF;
  std::error_code EC;
  auto StatsOS =
      std::make_unique<llvm::raw_fd_ostream>(StatsFilename, EC, FileFlags);
  if (EC) {
    WithColor::error(errs(), ToolName)
        << "Unable to open statistics file: " << EC.message() << "\n";
    return 1;
  }

  llvm::PrintStatisticsJSON(*StatsOS);
  return 0;
}
