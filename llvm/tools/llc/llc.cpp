//===-- llc.cpp - Implement the LLVM Native Code Generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the llc code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "NewPMDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PGOOptions.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <list>
#include <memory>
#include <optional>
using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;
static codegen::RegisterMTuneFlag MTF;
static codegen::RegisterSaveStatsFlag SSF;

#include "Opts.inc"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"

using namespace llvm;
using namespace llvm::opt;

namespace {

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

enum OptionFlags {
  LlcLocalOption = (1 << 0),
};

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class LlcOptTable : public GenericOptTable {
public:
  LlcOptTable()
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

} // end anonymous namespace

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
// TODO: These are static because the prior implementation using cl::opt
// defined these as static globals. It's probably neater to instead have these
// in some Config struct that gets passed around, but that can be done after
// the migration to using OptTable.
static std::string InputFilename = "-";
static std::vector<std::string> InstPrinterOptions;
static std::string InputLanguage = "";
static std::string OutputFilename = "";
static std::string SplitDwarfOutputFile = "";
static unsigned TimeCompilations = 1u;
static bool TimeTrace = false;
static unsigned TimeTraceGranularity = 500;
static std::string TimeTraceFile = "";
static std::string BinutilsVersion = "";
static bool PreserveComments = true;
static char OptLevel = '2';
static std::string TargetTriple = "";
static std::string SplitDwarfFile = "";
static bool NoVerify = false;
static bool VerifyEach = false;
static bool DisableSimplifyLibCalls = false;
static bool ShowMCEncoding = false;
static std::optional<unsigned> OutputAsmVariant;
static std::optional<bool> DwarfDirectory;
static bool AsmVerbose = true;
static bool CompileTwice = false;
static bool DiscardValueNames = false;
static bool PrintMIR2VecVocab = false;
static bool PrintMIR2Vec = false;
static std::vector<std::string> IncludeDirs;
static bool RemarksWithHotness = false;
static uint64_t RemarksHotnessThreshold = 0;
static std::string RemarksFilename = "";
static std::string RemarksPasses = "";
static std::string RemarksFormat = "yaml";
static bool EnableNewPassManager = false;
static std::string PassPipeline = "";

static std::vector<std::string> &getRunPassNames() {
  static std::vector<std::string> RunPassNames;
  return RunPassNames;
}

// PGO command line options
enum PGOKind {
  NoPGO,
  SampleUse,
};
static PGOKind PGOKindFlag = NoPGO;

// Function to set PGO options on TargetMachine based on command line flags.
static void setPGOOptions(TargetMachine &TM) {
  std::optional<PGOOptions> PGOOpt;

  switch (PGOKindFlag) {
  case SampleUse:
    // Use default values for other PGOOptions parameters. This parameter
    // is used to test that PGO data is preserved at -O0.
    PGOOpt = PGOOptions("", "", "", "", PGOOptions::SampleUse,
                        PGOOptions::NoCSAction);
    break;
  case NoPGO:
    PGOOpt = std::nullopt;
    break;
  }

  if (PGOOpt)
    TM.setPGOOption(PGOOpt);
}

static int compileModule(char **argv, SmallVectorImpl<PassPlugin> &,
                         LLVMContext &Context, std::string &OutputFilename);

[[noreturn]] static void reportError(Twine Msg, StringRef Filename = "") {
  SmallString<256> Prefix;
  if (!Filename.empty()) {
    if (Filename == "-")
      Filename = "<stdin>";
    ("'" + Twine(Filename) + "': ").toStringRef(Prefix);
  }
  WithColor::error(errs(), "llc") << Prefix << Msg << "\n";
  exit(1);
}

[[noreturn]] static void reportError(Error Err, StringRef Filename) {
  assert(Err);
  handleAllErrors(createFileError(Filename, std::move(Err)),
                  [&](const ErrorInfoBase &EI) { reportError(EI.message()); });
  llvm_unreachable("reportError() should not return");
}

static std::unique_ptr<ToolOutputFile> GetOutputStream(Triple::OSType OS) {
  // If we don't yet have an output filename, make one.
  if (OutputFilename.empty()) {
    if (InputFilename == "-")
      OutputFilename = "-";
    else {
      // If InputFilename ends in .bc or .ll, remove it.
      StringRef IFN = InputFilename;
      if (IFN.ends_with(".bc") || IFN.ends_with(".ll"))
        OutputFilename = std::string(IFN.drop_back(3));
      else if (IFN.ends_with(".mir"))
        OutputFilename = std::string(IFN.drop_back(4));
      else
        OutputFilename = std::string(IFN);

      switch (codegen::getFileType()) {
      case CodeGenFileType::AssemblyFile:
        OutputFilename += ".s";
        break;
      case CodeGenFileType::ObjectFile:
        if (OS == Triple::Win32)
          OutputFilename += ".obj";
        else
          OutputFilename += ".o";
        break;
      case CodeGenFileType::Null:
        OutputFilename = "-";
        break;
      }
    }
  }

  // Decide if we need "binary" output.
  bool Binary = false;
  switch (codegen::getFileType()) {
  case CodeGenFileType::AssemblyFile:
    break;
  case CodeGenFileType::ObjectFile:
  case CodeGenFileType::Null:
    Binary = true;
    break;
  }

  // Open the file.
  std::error_code EC;
  sys::fs::OpenFlags OpenFlags = sys::fs::OF_None;
  if (!Binary)
    OpenFlags |= sys::fs::OF_TextWithCRLF;
  auto FDOut = std::make_unique<ToolOutputFile>(OutputFilename, EC, OpenFlags);
  if (EC)
    reportError(EC.message());
  return FDOut;
}

static void RecordLlcOpts(opt::InputArgList &Args,
                          SmallVector<PassPlugin, 1> &PluginList) {
  if (Args.hasArg(OPT_version)) {
    // Register the Target and CPU printer for --version.
    cl::AddExtraVersionPrinter(sys::printDefaultTargetAndDetectedCPU);
    // Register the target printer for --version.
    cl::AddExtraVersionPrinter(
        TargetRegistry::printRegisteredTargetsForVersion);
    cl::PrintVersionMessage();
    exit(0);
  }

  for (const auto *A : Args) {
    if (A->getOption().getID() == OPT_INPUT)
      InputFilename = A->getValue();
  }

  InstPrinterOptions = Args.getAllArgValues(OPT_M);
  if (const opt::Arg *A = Args.getLastArg(OPT_x_EQ, OPT_x))
    InputLanguage = A->getValue();
  if (const opt::Arg *A = Args.getLastArg(OPT_o_EQ, OPT_o))
    OutputFilename = A->getValue();
  if (const opt::Arg *A =
          Args.getLastArg(OPT_split_dwarf_output_EQ, OPT_split_dwarf_output))
    SplitDwarfOutputFile = A->getValue();

  if (const opt::Arg *A =
          Args.getLastArg(OPT_time_compilations_EQ, OPT_time_compilations))
    StringRef(A->getValue()).getAsInteger(10, TimeCompilations);
  TimeTrace = Args.hasArg(OPT_time_trace);
  if (const opt::Arg *A = Args.getLastArg(OPT_time_trace_granularity_EQ,
                                          OPT_time_trace_granularity))
    StringRef(A->getValue()).getAsInteger(10, TimeTraceGranularity);
  if (const opt::Arg *A =
          Args.getLastArg(OPT_time_trace_file_EQ, OPT_time_trace_file))
    TimeTraceFile = A->getValue();

  if (const opt::Arg *A =
          Args.getLastArg(OPT_binutils_version_EQ, OPT_binutils_version))
    BinutilsVersion = A->getValue();

  if (const opt::Arg *A = Args.getLastArg(OPT_O_flag)) {
    StringRef Val = A->getValue();
    if (Val.starts_with("="))
      Val = Val.drop_front();
    if (!Val.empty())
      OptLevel = Val[0];
  }

  if (const opt::Arg *A = Args.getLastArg(OPT_mtriple_EQ, OPT_mtriple))
    TargetTriple = A->getValue();
  if (const opt::Arg *A =
          Args.getLastArg(OPT_split_dwarf_file_EQ, OPT_split_dwarf_file))
    SplitDwarfFile = A->getValue();

  NoVerify = Args.hasArg(OPT_disable_verify);
  VerifyEach = Args.hasArg(OPT_verify_each);
  DisableSimplifyLibCalls = Args.hasArg(OPT_disable_simplify_libcalls);
  ShowMCEncoding = Args.hasArg(OPT_show_mc_encoding);

  if (const opt::Arg *A =
          Args.getLastArg(OPT_output_asm_variant_EQ, OPT_output_asm_variant)) {
    unsigned variant;
    StringRef(A->getValue()).getAsInteger(10, variant);
    OutputAsmVariant = variant;
  }

  auto GetLastBooleanArg = [&Args](unsigned OptID,
                                   unsigned OptIDEQ) -> std::optional<bool> {
    if (const opt::Arg *A = Args.getLastArg(OptIDEQ)) {
      return StringRef(A->getValue()) == "1" ||
             StringRef(A->getValue()) == "true";
    } else if (Args.hasArg(OptID))
      return true;
    return {};
  };

  if (auto Val = GetLastBooleanArg(OPT_dwarf_directory, OPT_dwarf_directory_EQ))
    DwarfDirectory = *Val;

  if (auto Val = GetLastBooleanArg(OPT_asm_verbose, OPT_asm_verbose_EQ))
    AsmVerbose = *Val;

  if (const Arg *A = Args.getLastArg(OPT_preserve_as_comments))
    PreserveComments = A->getValue();

  CompileTwice = Args.hasArg(OPT_compile_twice);
  DiscardValueNames = Args.hasArg(OPT_discard_value_names);
  PrintMIR2VecVocab = Args.hasArg(OPT_print_mir2vec_vocab);
  PrintMIR2Vec = Args.hasArg(OPT_print_mir2vec);

  IncludeDirs = Args.getAllArgValues(OPT_I);
  for (std::string &Dir : IncludeDirs) {
    // Some of the values may include a '=' prefix, remove it.
    if (Dir.front() == '=')
      Dir = Dir.substr(1);
  }

  if (auto Val = GetLastBooleanArg(OPT_pass_remarks_with_hotness,
                                   OPT_pass_remarks_with_hotness_EQ))
    RemarksWithHotness = *Val;

  if (const opt::Arg *A = Args.getLastArg(OPT_pass_remarks_hotness_threshold_EQ,
                                          OPT_pass_remarks_hotness_threshold)) {
    if (StringRef(A->getValue()) != "auto") {
      uint64_t Val;
      StringRef(A->getValue()).getAsInteger(10, Val);
      RemarksHotnessThreshold = Val;
    }
  }
  if (const opt::Arg *A =
          Args.getLastArg(OPT_pass_remarks_output_EQ, OPT_pass_remarks_output))
    RemarksFilename = A->getValue();
  if (const opt::Arg *A =
          Args.getLastArg(OPT_pass_remarks_filter_EQ, OPT_pass_remarks_filter))
    RemarksPasses = A->getValue();
  if (const opt::Arg *A =
          Args.getLastArg(OPT_pass_remarks_format_EQ, OPT_pass_remarks_format))
    RemarksFormat = A->getValue();

  for (const std::string &PluginPath :
       Args.getAllArgValues(OPT_load_pass_plugin_EQ)) {
    auto Plugin = PassPlugin::Load(PluginPath);
    if (!Plugin)
      reportFatalUsageError(Plugin.takeError());
    PluginList.emplace_back(Plugin.get());
  }

  if (auto Val = GetLastBooleanArg(OPT_enable_new_pm, OPT_enable_new_pm_EQ))
    EnableNewPassManager = *Val;

  if (const opt::Arg *A = Args.getLastArg(OPT_passes_EQ, OPT_passes))
    PassPipeline = A->getValue();

  for (const auto *A : Args) {
    if (A->getOption().matches(OPT_run_pass) ||
        A->getOption().matches(OPT_run_pass_EQ)) {
      SmallVector<StringRef, 8> PassNames;
      StringRef(A->getValue()).split(PassNames, ',', -1, false);
      for (auto PassName : PassNames)
        getRunPassNames().push_back(std::string(PassName));
    }
  }

  if (const opt::Arg *A = Args.getLastArg(OPT_pgo_kind_EQ, OPT_pgo_kind)) {
    if (StringRef(A->getValue()) == "pgo-sample-use-pipeline")
      PGOKindFlag = SampleUse;
    else
      PGOKindFlag = NoPGO;
  }
}

// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeIPO(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
  initializeTransformUtils(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);

  SmallVector<PassPlugin, 1> PluginList;

  LlcOptTable Tbl;
  unsigned MissingArgIndex, MissingArgCount;
  ArrayRef<const char *> ArgsArr = ArrayRef(argv + 1, argc - 1);
  opt::InputArgList Args =
      Tbl.ParseArgs(ArgsArr, MissingArgIndex, MissingArgCount);

  if (MissingArgCount) {
    reportError("missing argument to option: " +
                Twine(Args.getArgString(MissingArgIndex)));
  }

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(), "llc [options] <input bitcode>",
                  "llvm system compiler");
    return 0;
  }

  RecordLlcOpts(Args, PluginList);

  // Arguments not consumed by llc directly must be passed
  // to the backend via cl::ParseCommandLineOptions. Unfortunately, this
  // function only accepts a raw argc + argv. Ideally, we'd just construct a
  // `const char*` vector that we could pass to this function consisting of
  // string pointers to the original argv from main, but there's no easy way to
  // do this. Here, we instead reconstruct the arguments as strings and convert
  // them to `const char*`.
  std::vector<const char *> NewArgv;
  NewArgv.push_back(argv[0]);

  // This is used as storage for arguments that need to be rendered and
  // eventually passed into NewArgv. A std::list is used because it guarantees
  // that additions to RenderedArgs can be made without invalidating any
  // pointers to existing elements. These pointers are what we store in NewArgv.
  std::list<std::string> RenderedArgs;

  // Forward all options that are NOT locally consumed and NOT unknown/input.
  for (const auto *A : Args) {
    unsigned ID = A->getOption().getID();
    if (ID == OPT_INPUT)
      continue;

    if (ID == OPT_UNKNOWN) {
      NewArgv.push_back(A->getSpelling().data());
      continue;
    }

    if (A->getOption().hasFlag(LlcLocalOption))
      continue;

    // Here we render the argument into a list of strings. Rendered arguments
    // expand the original argument into a list of strings that can be passed
    // to cl::ParseCommandLineOptions.
    opt::ArgStringList TempList;
    A->render(Args, TempList);
    for (auto *ArgStr : TempList) {
      StringRef S(ArgStr);
      if (S.starts_with("=")) {
        // If the argument starts with "=", it means it's a joined argument,
        // and we should append it to the last rendered argument.
        RenderedArgs.back().append(S.str());
        NewArgv.back() = RenderedArgs.back().c_str();
      } else if (!S.empty()) {
        // Otherwise, it's a new argument, and we should add it to the list.
        RenderedArgs.push_back(S.str());
        NewArgv.push_back(RenderedArgs.back().c_str());
      }
    }
  }

  // Now pass the filtered arguments to the backend.
  cl::ParseCommandLineOptions(NewArgv.size(), NewArgv.data(),
                              "llvm system compiler\n");

  if (!PassPipeline.empty() && !getRunPassNames().empty()) {
    errs() << "The `llc -run-pass=...` syntax for the new pass manager is "
              "not supported, please use `llc -passes=<pipeline>` (or the `-p` "
              "alias for a more concise version).\n";
    return 1;
  }

  if (TimeTrace)
    timeTraceProfilerInitialize(TimeTraceGranularity, argv[0]);
  llvm::scope_exit TimeTraceScopeExit([]() {
    if (TimeTrace) {
      if (auto E = timeTraceProfilerWrite(TimeTraceFile, OutputFilename)) {
        handleAllErrors(std::move(E), [&](const StringError &SE) {
          errs() << SE.getMessage() << "\n";
        });
        return;
      }
      timeTraceProfilerCleanup();
    }
  });

  LLVMContext Context;
  Context.setDiscardValueNames(DiscardValueNames);

  // Set a diagnostic handler that doesn't exit on the first error
  Context.setDiagnosticHandler(std::make_unique<LLCDiagnosticHandler>());

  Expected<LLVMRemarkFileHandle> RemarksFileOrErr =
      setupLLVMOptimizationRemarks(Context, RemarksFilename, RemarksPasses,
                                   RemarksFormat, RemarksWithHotness,
                                   RemarksHotnessThreshold);
  if (Error E = RemarksFileOrErr.takeError())
    reportError(std::move(E), RemarksFilename);
  LLVMRemarkFileHandle RemarksFile = std::move(*RemarksFileOrErr);

  codegen::MaybeEnableStatistics();
  std::string OutputFilename;

  if (InputLanguage != "" && InputLanguage != "ir" && InputLanguage != "mir")
    reportError("input language must be '', 'IR' or 'MIR'");

  // Compile the module TimeCompilations times to give better compile time
  // metrics.
  for (unsigned I = TimeCompilations; I; --I)
    if (int RetVal = compileModule(argv, PluginList, Context, OutputFilename))
      return RetVal;

  if (RemarksFile)
    RemarksFile->keep();

  return codegen::MaybeSaveStatistics(OutputFilename, "llc");
}

static bool addPass(PassManagerBase &PM, const char *argv0, StringRef PassName,
                    TargetPassConfig &TPC) {
  if (PassName == "none")
    return false;

  const PassRegistry *PR = PassRegistry::getPassRegistry();
  const PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    WithColor::error(errs(), argv0)
        << "run-pass " << PassName << " is not registered.\n";
    return true;
  }

  Pass *P;
  if (PI->getNormalCtor())
    P = PI->getNormalCtor()();
  else {
    WithColor::error(errs(), argv0)
        << "cannot create pass: " << PI->getPassName() << "\n";
    return true;
  }
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  TPC.addMachinePrePasses();
  PM.add(P);
  TPC.addMachinePostPasses(Banner);

  return false;
}

static int compileModule(char **argv, SmallVectorImpl<PassPlugin> &PluginList,
                         LLVMContext &Context, std::string &OutputFilename) {
  // Load the module to be compiled...
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;
  Triple TheTriple;
  std::string CPUStr = codegen::getCPUStr();
  std::string TuneCPUStr = codegen::getTuneCPUStr();
  std::string FeaturesStr = codegen::getFeaturesStr();

  // Set attributes on functions as loaded from MIR from command line arguments.
  auto setMIRFunctionAttributes = [&CPUStr, &TuneCPUStr,
                                   &FeaturesStr](Function &F) {
    codegen::setFunctionAttributes(F, CPUStr, FeaturesStr, TuneCPUStr);
  };

  CodeGenOptLevel OLvl;
  if (auto Level = CodeGenOpt::parseLevel(OptLevel)) {
    OLvl = *Level;
  } else {
    WithColor::error(errs(), argv[0]) << "invalid optimization level.\n";
    return 1;
  }

  // Parse 'none' or '$major.$minor'. Disallow -binutils-version=0 because we
  // use that to indicate the MC default.
  if (!BinutilsVersion.empty() && BinutilsVersion != "none") {
    StringRef V = BinutilsVersion;
    unsigned Num;
    if (V.consumeInteger(10, Num) || Num == 0 ||
        !(V.empty() ||
          (V.consume_front(".") && !V.consumeInteger(10, Num) && V.empty()))) {
      WithColor::error(errs(), argv[0])
          << "invalid -binutils-version, accepting 'none' or major.minor\n";
      return 1;
    }
  }
  TargetOptions Options;
  auto InitializeOptions = [&](const Triple &TheTriple) {
    Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

    if (Options.XCOFFReadOnlyPointers) {
      if (!TheTriple.isOSAIX())
        reportError("-mxcoff-roptr option is only supported on AIX",
                    InputFilename);

      // Since the storage mapping class is specified per csect,
      // without using data sections, it is less effective to use read-only
      // pointers. Using read-only pointers may cause other RO variables in the
      // same csect to become RW when the linker acts upon `-bforceimprw`;
      // therefore, we require that separate data sections are used in the
      // presence of ReadOnlyPointers. We respect the setting of data-sections
      // since we have not found reasons to do otherwise that overcome the user
      // surprise of not respecting the setting.
      if (!Options.DataSections)
        reportError("-mxcoff-roptr option must be used with -data-sections",
                    InputFilename);
    }

    if (TheTriple.isX86() &&
        codegen::getFuseFPOps() != FPOpFusion::FPOpFusionMode::Standard)
      WithColor::warning(errs(), argv[0])
          << "X86 backend ignores --fp-contract setting; use IR fast-math "
             "flags instead.";

    Options.BinutilsVersion =
        TargetMachine::parseBinutilsVersion(BinutilsVersion);
    Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
    Options.MCOptions.AsmVerbose = AsmVerbose;
    Options.MCOptions.PreserveAsmComments = PreserveComments;
    if (OutputAsmVariant.has_value())
      Options.MCOptions.OutputAsmVariant = *OutputAsmVariant;
    Options.MCOptions.IASSearchPaths = IncludeDirs;

    Options.MCOptions.InstPrinterOptions = InstPrinterOptions;
    Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
    if (DwarfDirectory.has_value()) {
      Options.MCOptions.MCUseDwarfDirectory =
          *DwarfDirectory ? MCTargetOptions::EnableDwarfDirectory
                          : MCTargetOptions::DisableDwarfDirectory;
    } else {
      // -dwarf-directory is not set explicitly. Some assemblers
      // (e.g. GNU as or ptxas) do not support `.file directory'
      // syntax prior to DWARFv5. Let the target decide the default
      // value.
      Options.MCOptions.MCUseDwarfDirectory =
          MCTargetOptions::DefaultDwarfDirectory;
    }
  };

  std::optional<Reloc::Model> RM = codegen::getExplicitRelocModel();
  std::optional<CodeModel::Model> CM = codegen::getExplicitCodeModel();

  const Target *TheTarget = nullptr;
  std::unique_ptr<TargetMachine> Target;

  // If user just wants to list available options, skip module loading
  auto MAttrs = codegen::getMAttrs();
  bool SkipModule =
      CPUStr == "help" || TuneCPUStr == "help" || is_contained(MAttrs, "help");
  if (SkipModule) {
    if (!TargetTriple.empty())
      TheTriple = Triple(Triple::normalize(TargetTriple));
    else
      TheTriple = Triple(sys::getDefaultTargetTriple());

    // Get the target specific parser.
    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error << "\n";
      return 1;
    }

    InitializeOptions(TheTriple);
    // Pass "help" as CPU for -mtune=help
    std::string SkipModuleCPU = (TuneCPUStr == "help" ? "help" : CPUStr);
    // Create the target machine just to print the help info. Use unique_ptr
    // to avoid a memory leak.
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple, SkipModuleCPU, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");

    // If we don't have a module then just exit now. We do this down
    // here since the CPU/Feature help is underneath the target machine
    // creation.
    return 0;
  }

  auto SetDataLayout = [&](StringRef DataLayoutTargetTriple,
                           StringRef OldDLStr) -> std::optional<std::string> {
    // If we are supposed to override the target triple, do so now.
    std::string IRTargetTriple = DataLayoutTargetTriple.str();
    if (!TargetTriple.empty())
      IRTargetTriple = Triple::normalize(TargetTriple);
    TheTriple = Triple(IRTargetTriple);
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error << "\n";
      exit(1);
    }

    InitializeOptions(TheTriple);
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");

    // Set PGO options based on command line flags
    setPGOOptions(*Target);

    return Target->createDataLayout().getStringRepresentation();
  };
  if (InputLanguage == "mir" ||
      (InputLanguage == "" && StringRef(InputFilename).ends_with(".mir"))) {
    MIR = createMIRParserFromFile(InputFilename, Err, Context,
                                  setMIRFunctionAttributes);
    if (MIR)
      M = MIR->parseIRModule(SetDataLayout);
  } else {
    M = parseIRFile(InputFilename, Err, Context,
                    ParserCallbacks(SetDataLayout));
  }
  if (!M) {
    Err.print(argv[0], WithColor::error(errs(), argv[0]));
    return 1;
  }
  if (!TargetTriple.empty())
    M->setTargetTriple(Triple(Triple::normalize(TargetTriple)));

  std::optional<CodeModel::Model> CM_IR = M->getCodeModel();
  if (!CM && CM_IR)
    Target->setCodeModel(*CM_IR);
  if (std::optional<uint64_t> LDT = codegen::getExplicitLargeDataThreshold())
    Target->setLargeDataThreshold(*LDT);

  if (codegen::getFloatABIForCalls() != FloatABI::Default)
    Target->Options.FloatABIType = codegen::getFloatABIForCalls();

  // Figure out where we are going to send the output.
  std::unique_ptr<ToolOutputFile> Out = GetOutputStream(TheTriple.getOS());
  if (!Out)
    return 1;

  // Ensure the filename is passed down to CodeViewDebug.
  Target->Options.ObjectFilenameForDebug = Out->outputFilename();

  // Return a copy of the output filename via the output param
  OutputFilename = Out->outputFilename();

  // Tell target that this tool is not necessarily used with argument ABI
  // compliance (i.e. narrow integer argument extensions).
  Target->Options.VerifyArgABICompliance = 0;

  std::unique_ptr<ToolOutputFile> DwoOut;
  if (!SplitDwarfOutputFile.empty()) {
    std::error_code EC;
    DwoOut = std::make_unique<ToolOutputFile>(SplitDwarfOutputFile, EC,
                                              sys::fs::OF_None);
    if (EC)
      reportError(EC.message(), SplitDwarfOutputFile);
  }

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(M->getTargetTriple(), Target->Options.VecLib);

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();

  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (!NoVerify && verifyModule(*M, &errs()))
    reportError("input module cannot be verified", InputFilename);

  // Override function attributes based on CPUStr, TuneCPUStr, FeaturesStr, and
  // command line flags.
  codegen::setFunctionAttributes(*M, CPUStr, FeaturesStr, TuneCPUStr);

  for (auto &Plugin : PluginList) {
    CodeGenFileType CGFT = codegen::getFileType();
    if (Plugin.invokePreCodeGenCallback(*M, *Target, CGFT, Out->os())) {
      // TODO: Deduplicate code with below and the NewPMDriver.
      if (Context.getDiagHandlerPtr()->HasErrors)
        exit(1);
      Out->keep();
      return 0;
    }
  }

  if (mc::getExplicitRelaxAll() &&
      codegen::getFileType() != CodeGenFileType::ObjectFile)
    WithColor::warning(errs(), argv[0])
        << ": warning: ignoring -mc-relax-all because filetype != obj";

  VerifierKind VK = VerifierKind::InputOutput;
  if (NoVerify)
    VK = VerifierKind::None;
  else if (VerifyEach)
    VK = VerifierKind::EachPass;

  if (EnableNewPassManager || !PassPipeline.empty()) {
    return compileModuleWithNewPM(argv[0], std::move(M), std::move(MIR),
                                  std::move(Target), std::move(Out),
                                  std::move(DwoOut), Context, TLII, VK,
                                  PassPipeline, codegen::getFileType());
  }

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  PM.add(new RuntimeLibraryInfoWrapper(
      TheTriple, Target->Options.ExceptionModel, Target->Options.FloatABIType,
      Target->Options.EABIVersion, Options.MCOptions.ABIName,
      Target->Options.VecLib));

  {
    raw_pwrite_stream *OS = &Out->os();

    // Manually do the buffering rather than using buffer_ostream,
    // so we can memcmp the contents in CompileTwice mode
    SmallVector<char, 0> Buffer;
    std::unique_ptr<raw_svector_ostream> BOS;
    if ((codegen::getFileType() != CodeGenFileType::AssemblyFile &&
         !Out->os().supportsSeeking()) ||
        CompileTwice) {
      BOS = std::make_unique<raw_svector_ostream>(Buffer);
      OS = BOS.get();
    }

    const char *argv0 = argv[0];
    MachineModuleInfoWrapperPass *MMIWP =
        new MachineModuleInfoWrapperPass(Target.get());

    // Set a temporary diagnostic handler. This is used before
    // MachineModuleInfoWrapperPass::doInitialization for features like -M.
    bool HasMCErrors = false;
    MCContext &MCCtx = MMIWP->getMMI().getContext();
    MCCtx.setDiagnosticHandler([&](const SMDiagnostic &SMD, bool IsInlineAsm,
                                   const SourceMgr &SrcMgr,
                                   std::vector<const MDNode *> &LocInfos) {
      WithColor::error(errs(), argv0) << SMD.getMessage() << '\n';
      HasMCErrors = true;
    });

    // Construct a custom pass pipeline that starts after instruction
    // selection.
    if (!getRunPassNames().empty()) {
      if (!MIR) {
        WithColor::error(errs(), argv[0])
            << "run-pass is for .mir file only.\n";
        delete MMIWP;
        return 1;
      }
      TargetPassConfig *PTPC = Target->createPassConfig(PM);
      TargetPassConfig &TPC = *PTPC;
      if (TPC.hasLimitedCodeGenPipeline()) {
        WithColor::error(errs(), argv[0])
            << "run-pass cannot be used with "
            << TPC.getLimitedCodeGenPipelineReason() << ".\n";
        delete PTPC;
        delete MMIWP;
        return 1;
      }

      TPC.setDisableVerify(NoVerify);
      PM.add(&TPC);
      PM.add(MMIWP);
      TPC.printAndVerify("");
      for (const std::string &RunPassName : getRunPassNames()) {
        if (addPass(PM, argv0, RunPassName, TPC))
          return 1;
      }
      TPC.setInitialized();
      PM.add(createPrintMIRPass(*OS));

      // Add MIR2Vec vocabulary printer if requested
      if (PrintMIR2VecVocab) {
        PM.add(createMIR2VecVocabPrinterLegacyPass(errs()));
      }

      // Add MIR2Vec printer if requested
      if (PrintMIR2Vec) {
        PM.add(createMIR2VecPrinterLegacyPass(errs()));
      }

      PM.add(createFreeMachineFunctionPass());
    } else {
      if (Target->addPassesToEmitFile(PM, *OS, DwoOut ? &DwoOut->os() : nullptr,
                                      codegen::getFileType(), NoVerify,
                                      MMIWP)) {
        if (!HasMCErrors)
          reportError("target does not support generation of this file type");
      }

      // Add MIR2Vec vocabulary printer if requested
      if (PrintMIR2VecVocab) {
        PM.add(createMIR2VecVocabPrinterLegacyPass(errs()));
      }

      // Add MIR2Vec printer if requested
      if (PrintMIR2Vec) {
        PM.add(createMIR2VecPrinterLegacyPass(errs()));
      }
    }

    Target->getObjFileLowering()->Initialize(MMIWP->getMMI().getContext(),
                                             *Target);
    if (MIR) {
      assert(MMIWP && "Forgot to create MMIWP?");
      if (MIR->parseMachineFunctions(*M, MMIWP->getMMI()))
        return 1;
    }

    // Before executing passes, print the final values of the LLVM options.
    cl::PrintOptionValues();

    // If requested, run the pass manager over the same module again,
    // to catch any bugs due to persistent state in the passes. Note that
    // opt has the same functionality, so it may be worth abstracting this out
    // in the future.
    SmallVector<char, 0> CompileTwiceBuffer;
    if (CompileTwice) {
      std::unique_ptr<Module> M2(llvm::CloneModule(*M));
      PM.run(*M2);
      CompileTwiceBuffer = Buffer;
      Buffer.clear();
    }

    PM.run(*M);

    if (Context.getDiagHandlerPtr()->HasErrors || HasMCErrors)
      return 1;

    // Compare the two outputs and make sure they're the same
    if (CompileTwice) {
      if (Buffer.size() != CompileTwiceBuffer.size() ||
          (memcmp(Buffer.data(), CompileTwiceBuffer.data(), Buffer.size()) !=
           0)) {
        errs()
            << "Running the pass manager twice changed the output.\n"
               "Writing the result of the second run to the specified output\n"
               "To generate the one-run comparison binary, just run without\n"
               "the compile-twice option\n";
        Out->os() << Buffer;
        Out->keep();
        return 1;
      }
    }

    if (BOS) {
      Out->os() << Buffer;
    }
  }

  // Declare success.
  Out->keep();
  if (DwoOut)
    DwoOut->keep();

  return 0;
}
