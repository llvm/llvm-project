//===- opt.cpp - The LLVM Modular Optimizer -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimizations may be specified an arbitrary number of times on the command
// line, They are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "BreakpointPrinter.h"
#include "NewPMDriver.h"
#include "PassPrinters.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Coroutines.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Debugify.h"
#include <algorithm>
#include <memory>
using namespace llvm;
using namespace opt_tool;

static codegen::RegisterCodeGenFlags CFG;

// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Optimizations available:"));

static cl::opt<bool> EnableNewPassManager(
    "enable-new-pm", cl::desc("Enable the new pass manager"), cl::init(false));

// This flag specifies a textual description of the optimization pass pipeline
// to run over the module. This flag switches opt to use the new pass manager
// infrastructure, completely disabling all of the flags specific to the old
// pass management.
static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc("A textual description of the pass pipeline for optimizing"),
    cl::Hidden);

// Other command line options...
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode file>"),
    cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
PrintEachXForm("p", cl::desc("Print module after each transformation"));

static cl::opt<bool>
NoOutput("disable-output",
         cl::desc("Do not write result bitcode file"), cl::Hidden);

static cl::opt<bool>
OutputAssembly("S", cl::desc("Write output as LLVM assembly"));

static cl::opt<bool>
    OutputThinLTOBC("thinlto-bc",
                    cl::desc("Write output as ThinLTO-ready bitcode"));

static cl::opt<bool>
    SplitLTOUnit("thinlto-split-lto-unit",
                 cl::desc("Enable splitting of a ThinLTO LTOUnit"));

static cl::opt<std::string> ThinLinkBitcodeFile(
    "thin-link-bitcode-file", cl::value_desc("filename"),
    cl::desc(
        "A file in which to write minimized bitcode for the thin link only"));

static cl::opt<bool>
NoVerify("disable-verify", cl::desc("Do not run the verifier"), cl::Hidden);

static cl::opt<bool> NoUpgradeDebugInfo("disable-upgrade-debug-info",
                                        cl::desc("Generate invalid output"),
                                        cl::ReallyHidden);

static cl::opt<bool> VerifyEach("verify-each",
                                cl::desc("Verify after each transform"));

static cl::opt<bool>
    DisableDITypeMap("disable-debug-info-type-map",
                     cl::desc("Don't use a uniquing type map for debug info"));

static cl::opt<bool>
StripDebug("strip-debug",
           cl::desc("Strip debugger symbol info from translation unit"));

static cl::opt<bool>
    StripNamedMetadata("strip-named-metadata",
                       cl::desc("Strip module-level named metadata"));

static cl::opt<bool> DisableInline("disable-inlining",
                                   cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableOptimizations("disable-opt",
                     cl::desc("Do not run any optimization passes"));

static cl::opt<bool>
StandardLinkOpts("std-link-opts",
                 cl::desc("Include the standard link time optimizations"));

static cl::opt<bool>
OptLevelO0("O0",
  cl::desc("Optimization level 0. Similar to clang -O0"));

static cl::opt<bool>
OptLevelO1("O1",
           cl::desc("Optimization level 1. Similar to clang -O1"));

static cl::opt<bool>
OptLevelO2("O2",
           cl::desc("Optimization level 2. Similar to clang -O2"));

static cl::opt<bool>
OptLevelOs("Os",
           cl::desc("Like -O2 with extra optimizations for size. Similar to clang -Os"));

static cl::opt<bool>
OptLevelOz("Oz",
           cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"));

static cl::opt<bool>
OptLevelO3("O3",
           cl::desc("Optimization level 3. Similar to clang -O3"));

static cl::opt<unsigned>
CodeGenOptLevel("codegen-opt-level",
                cl::desc("Override optimization level for codegen hooks"));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

cl::opt<bool> DisableLoopUnrolling(
    "disable-loop-unrolling",
    cl::desc("Disable loop unrolling in all relevant passes"), cl::init(false));

static cl::opt<bool> EmitSummaryIndex("module-summary",
                                      cl::desc("Emit module summary index"),
                                      cl::init(false));

static cl::opt<bool> EmitModuleHash("module-hash", cl::desc("Emit module hash"),
                                    cl::init(false));

static cl::opt<bool>
DisableSimplifyLibCalls("disable-simplify-libcalls",
                        cl::desc("Disable simplify-libcalls"));

static cl::list<std::string>
DisableBuiltins("disable-builtin",
                cl::desc("Disable specific target library builtin function"),
                cl::ZeroOrMore);

static cl::opt<bool>
AnalyzeOnly("analyze", cl::desc("Only perform analysis, no optimization"));

static cl::opt<bool> EnableDebugify(
    "enable-debugify",
    cl::desc(
        "Start the pipeline with debugify and end it with check-debugify"));

static cl::opt<bool> DebugifyEach(
    "debugify-each",
    cl::desc(
        "Start each pass with debugify and end it with check-debugify"));

static cl::opt<std::string>
    DebugifyExport("debugify-export",
                   cl::desc("Export per-pass debugify statistics to this file"),
                   cl::value_desc("filename"), cl::init(""));

static cl::opt<bool>
PrintBreakpoints("print-breakpoints-for-testing",
                 cl::desc("Print select breakpoints location for testing"));

static cl::opt<std::string> ClDataLayout("data-layout",
                                         cl::desc("data layout string to use"),
                                         cl::value_desc("layout-string"),
                                         cl::init(""));

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    RunTwice("run-twice",
             cl::desc("Run all passes twice, re-using the same pass manager."),
             cl::init(false), cl::Hidden);

static cl::opt<bool> DiscardValueNames(
    "discard-value-names",
    cl::desc("Discard names from Value (other than GlobalValue)."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> Coroutines(
  "enable-coroutines",
  cl::desc("Enable coroutine passes."),
  cl::init(false), cl::Hidden);

static cl::opt<bool> TimeTrace(
    "time-trace",
    cl::desc("Record time trace"));

static cl::opt<unsigned> TimeTraceGranularity(
    "time-trace-granularity",
    cl::desc("Minimum time granularity (in microseconds) traced by time profiler"),
    cl::init(500), cl::Hidden);

static cl::opt<std::string>
    TimeTraceFile("time-trace-file",
                    cl::desc("Specify time trace file destination"),
                    cl::value_desc("filename"));

static cl::opt<bool> RemarksWithHotness(
    "pass-remarks-with-hotness",
    cl::desc("With PGO, include profile count in optimization remarks"),
    cl::Hidden);

static cl::opt<unsigned>
    RemarksHotnessThreshold("pass-remarks-hotness-threshold",
                            cl::desc("Minimum profile count required for "
                                     "an optimization remark to be output"),
                            cl::Hidden);

static cl::opt<std::string>
    RemarksFilename("pass-remarks-output",
                    cl::desc("Output filename for pass remarks"),
                    cl::value_desc("filename"));

static cl::opt<std::string>
    RemarksPasses("pass-remarks-filter",
                  cl::desc("Only record optimization remarks from passes whose "
                           "names match the given regular expression"),
                  cl::value_desc("regex"));

static cl::opt<std::string> RemarksFormat(
    "pass-remarks-format",
    cl::desc("The format used for serializing remarks (default: YAML)"),
    cl::value_desc("format"), cl::init("yaml"));

cl::opt<PGOKind>
    PGOKindFlag("pgo-kind", cl::init(NoPGO), cl::Hidden,
                cl::desc("The kind of profile guided optimization"),
                cl::values(clEnumValN(NoPGO, "nopgo", "Do not use PGO."),
                           clEnumValN(InstrGen, "pgo-instr-gen-pipeline",
                                      "Instrument the IR to generate profile."),
                           clEnumValN(InstrUse, "pgo-instr-use-pipeline",
                                      "Use instrumented profile to guide PGO."),
                           clEnumValN(SampleUse, "pgo-sample-use-pipeline",
                                      "Use sampled profile to guide PGO.")));
cl::opt<std::string> ProfileFile("profile-file",
                                 cl::desc("Path to the profile."), cl::Hidden);

cl::opt<CSPGOKind> CSPGOKindFlag(
    "cspgo-kind", cl::init(NoCSPGO), cl::Hidden,
    cl::desc("The kind of context sensitive profile guided optimization"),
    cl::values(
        clEnumValN(NoCSPGO, "nocspgo", "Do not use CSPGO."),
        clEnumValN(
            CSInstrGen, "cspgo-instr-gen-pipeline",
            "Instrument (context sensitive) the IR to generate profile."),
        clEnumValN(
            CSInstrUse, "cspgo-instr-use-pipeline",
            "Use instrumented (context sensitive) profile to guide PGO.")));
cl::opt<std::string> CSProfileGenFile(
    "cs-profilegen-file",
    cl::desc("Path to the instrumented context sensitive profile."),
    cl::Hidden);

static inline void addPass(legacy::PassManagerBase &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);

  // If we are verifying all of the intermediate steps, add the verifier...
  if (VerifyEach)
    PM.add(createVerifierPass());
}

/// This routine adds optimization passes based on selected optimization level,
/// OptLevel.
///
/// OptLevel - Optimization Level
static void AddOptimizationPasses(legacy::PassManagerBase &MPM,
                                  legacy::FunctionPassManager &FPM,
                                  TargetMachine *TM, unsigned OptLevel,
                                  unsigned SizeLevel) {
  if (!NoVerify || VerifyEach)
    FPM.add(createVerifierPass()); // Verify that input is correct

  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;

  if (DisableInline) {
    // No inlining pass
  } else if (OptLevel > 1) {
    Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel, false);
  } else {
    Builder.Inliner = createAlwaysInlinerLegacyPass();
  }
  Builder.DisableUnrollLoops = (DisableLoopUnrolling.getNumOccurrences() > 0) ?
                               DisableLoopUnrolling : OptLevel == 0;

  Builder.LoopVectorize = OptLevel > 1 && SizeLevel < 2;

  Builder.SLPVectorize = OptLevel > 1 && SizeLevel < 2;

  if (TM)
    TM->adjustPassManager(Builder);

  if (Coroutines)
    addCoroutinePassesToExtensionPoints(Builder);

  switch (PGOKindFlag) {
  case InstrGen:
    Builder.EnablePGOInstrGen = true;
    Builder.PGOInstrGen = ProfileFile;
    break;
  case InstrUse:
    Builder.PGOInstrUse = ProfileFile;
    break;
  case SampleUse:
    Builder.PGOSampleUse = ProfileFile;
    break;
  default:
    break;
  }

  switch (CSPGOKindFlag) {
  case CSInstrGen:
    Builder.EnablePGOCSInstrGen = true;
    break;
  case CSInstrUse:
    Builder.EnablePGOCSInstrUse = true;
    break;
  default:
    break;
  }

  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(MPM);
}

static void AddStandardLinkPasses(legacy::PassManagerBase &PM) {
  PassManagerBuilder Builder;
  Builder.VerifyInput = true;
  if (DisableOptimizations)
    Builder.OptLevel = 0;

  if (!DisableInline)
    Builder.Inliner = createFunctionInliningPass();
  Builder.populateLTOPassManager(PM);
}

//===----------------------------------------------------------------------===//
// CodeGen-related helper functions.
//

static CodeGenOpt::Level GetCodeGenOptLevel() {
  if (CodeGenOptLevel.getNumOccurrences())
    return static_cast<CodeGenOpt::Level>(unsigned(CodeGenOptLevel));
  if (OptLevelO1)
    return CodeGenOpt::Less;
  if (OptLevelO2)
    return CodeGenOpt::Default;
  if (OptLevelO3)
    return CodeGenOpt::Aggressive;
  return CodeGenOpt::None;
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(Triple TheTriple, StringRef CPUStr,
                                       StringRef FeaturesStr,
                                       const TargetOptions &Options) {
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return nullptr;
  }

  return TheTarget->createTargetMachine(
      TheTriple.getTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(),
      Options, codegen::getExplicitRelocModel(),
      codegen::getExplicitCodeModel(), GetCodeGenOptLevel());
}

#ifdef BUILD_EXAMPLES
void initializeExampleIRTransforms(llvm::PassRegistry &Registry);
#endif

static void exportDebugifyStats(llvm::StringRef Path,
                                const DebugifyStatsMap &Map) {
  std::error_code EC;
  raw_fd_ostream OS{Path, EC};
  if (EC) {
    errs() << "Could not open file: " << EC.message() << ", " << Path << '\n';
    return;
  }

  OS << "Pass Name" << ',' << "# of missing debug values" << ','
     << "# of missing locations" << ',' << "Missing/Expected value ratio" << ','
     << "Missing/Expected location ratio" << '\n';
  for (const auto &Entry : Map) {
    StringRef Pass = Entry.first;
    DebugifyStatistics Stats = Entry.second;

    OS << Pass << ',' << Stats.NumDbgValuesMissing << ','
       << Stats.NumDbgLocsMissing << ',' << Stats.getMissingValueRatio() << ','
       << Stats.getEmptyLocationRatio() << '\n';
  }
}

struct TimeTracerRAII {
  TimeTracerRAII(StringRef ProgramName) {
    if (TimeTrace)
      timeTraceProfilerInitialize(TimeTraceGranularity, ProgramName);
  }
  ~TimeTracerRAII() {
    if (TimeTrace) {
      if (auto E = timeTraceProfilerWrite(TimeTraceFile, OutputFilename)) {
        handleAllErrors(std::move(E), [&](const StringError &SE) {
          errs() << SE.getMessage() << "\n";
        });
        return;
      }
      timeTraceProfilerCleanup();
    }
  }
};

// For use in NPM transition.
// TODO: use a codegen version of PassRegistry.def/PassBuilder::is*Pass() once
// it exists.
static bool IsCodegenPass(StringRef Pass) {
  std::vector<StringRef> PassNamePrefix = {
      "x86-",  "xcore-", "wasm-",    "systemz-", "ppc-",   "nvvm-",   "nvptx-",
      "mips-", "lanai-", "hexagon-", "bpf-",     "avr-",   "thumb2-", "arm-",
      "si-",   "gcn-",   "amdgpu-",  "aarch64-", "amdgcn-"};
  std::vector<StringRef> PassNameContain = {"ehprepare"};
  std::vector<StringRef> PassNameExact = {
      "safe-stack",           "cost-model",
      "codegenprepare",       "interleaved-load-combine",
      "unreachableblockelim", "scalarize-masked-mem-intrin",
      "verify-safepoint-ir",  "divergence",
      "infer-address-spaces", "atomic-expand",
      "hardware-loops",       "type-promotion",
      "mve-tail-predication", "interleaved-access",
      "global-merge",         "pre-isel-intrinsic-lowering",
      "expand-reductions",    "indirectbr-expand",
      "generic-to-nvvm",      "expandmemcmp"};
  for (const auto &P : PassNamePrefix)
    if (Pass.startswith(P))
      return true;
  for (const auto &P : PassNameContain)
    if (Pass.contains(P))
      return true;
  for (const auto &P : PassNameExact)
    if (Pass == P)
      return true;
  return false;
}

// For use in NPM transition.
static bool CodegenPassSpecifiedInPassList() {
  for (const auto &P : PassList) {
    StringRef Arg = P->getPassArgument();
    if (IsCodegenPass(Arg))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  LLVMContext Context;

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeCoroutines(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeAggressiveInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeExpandMemCmpPassPass(Registry);
  initializeScalarizeMaskedMemIntrinPass(Registry);
  initializeCodeGenPreparePass(Registry);
  initializeAtomicExpandPass(Registry);
  initializeRewriteSymbolsLegacyPassPass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeDwarfEHPreparePass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeSjLjEHPreparePass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeGlobalMergePass(Registry);
  initializeIndirectBrExpandPassPass(Registry);
  initializeInterleavedLoadCombinePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeEntryExitInstrumenterPass(Registry);
  initializePostInlineEntryExitInstrumenterPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeExpandReductionsPass(Registry);
  initializeWasmEHPreparePass(Registry);
  initializeWriteBitcodePassPass(Registry);
  initializeHardwareLoopsPass(Registry);
  initializeTypePromotionPass(Registry);

#ifdef BUILD_EXAMPLES
  initializeExampleIRTransforms(Registry);
#endif

  cl::ParseCommandLineOptions(argc, argv,
    "llvm .bc -> .bc modular optimizer and analysis printer\n");

  if (AnalyzeOnly && NoOutput) {
    errs() << argv[0] << ": analyze mode conflicts with no-output mode.\n";
    return 1;
  }

  TimeTracerRAII TimeTracer(argv[0]);

  SMDiagnostic Err;

  Context.setDiscardValueNames(DiscardValueNames);
  if (!DisableDITypeMap)
    Context.enableDebugTypeODRUniquing();

  Expected<std::unique_ptr<ToolOutputFile>> RemarksFileOrErr =
      setupLLVMOptimizationRemarks(Context, RemarksFilename, RemarksPasses,
                                   RemarksFormat, RemarksWithHotness,
                                   RemarksHotnessThreshold);
  if (Error E = RemarksFileOrErr.takeError()) {
    errs() << toString(std::move(E)) << '\n';
    return 1;
  }
  std::unique_ptr<ToolOutputFile> RemarksFile = std::move(*RemarksFileOrErr);

  // Load the input module...
  auto SetDataLayout = [](StringRef) -> Optional<std::string> {
    if (ClDataLayout.empty())
      return None;
    return ClDataLayout;
  };
  std::unique_ptr<Module> M;
  if (NoUpgradeDebugInfo)
    M = parseAssemblyFileWithIndexNoUpgradeDebugInfo(
            InputFilename, Err, Context, nullptr, SetDataLayout)
            .Mod;
  else
    M = parseIRFile(InputFilename, Err, Context, SetDataLayout);

  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Strip debug info before running the verifier.
  if (StripDebug)
    StripDebugInfo(*M);

  // Erase module-level named metadata, if requested.
  if (StripNamedMetadata) {
    while (!M->named_metadata_empty()) {
      NamedMDNode *NMD = &*M->named_metadata_begin();
      M->eraseNamedMetadata(NMD);
    }
  }

  // If we are supposed to override the target triple or data layout, do so now.
  if (!TargetTriple.empty())
    M->setTargetTriple(Triple::normalize(TargetTriple));

  // Immediately run the verifier to catch any problems before starting up the
  // pass pipelines.  Otherwise we can crash on broken code during
  // doInitialization().
  if (!NoVerify && verifyModule(*M, &errs())) {
    errs() << argv[0] << ": " << InputFilename
           << ": error: input module is broken!\n";
    return 1;
  }

  // Enable testing of whole program devirtualization on this module by invoking
  // the facility for updating public visibility to linkage unit visibility when
  // specified by an internal option. This is normally done during LTO which is
  // not performed via opt.
  updateVCallVisibilityInModule(*M,
                                /* WholeProgramVisibilityEnabledInLTO */ false);

  // Figure out what stream we are supposed to write to...
  std::unique_ptr<ToolOutputFile> Out;
  std::unique_ptr<ToolOutputFile> ThinLinkOut;
  if (NoOutput) {
    if (!OutputFilename.empty())
      errs() << "WARNING: The -o (output filename) option is ignored when\n"
                "the --disable-output option is used.\n";
  } else {
    // Default to standard output.
    if (OutputFilename.empty())
      OutputFilename = "-";

    std::error_code EC;
    sys::fs::OpenFlags Flags = OutputAssembly ? sys::fs::OF_Text
                                              : sys::fs::OF_None;
    Out.reset(new ToolOutputFile(OutputFilename, EC, Flags));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }

    if (!ThinLinkBitcodeFile.empty()) {
      ThinLinkOut.reset(
          new ToolOutputFile(ThinLinkBitcodeFile, EC, sys::fs::OF_None));
      if (EC) {
        errs() << EC.message() << '\n';
        return 1;
      }
    }
  }

  Triple ModuleTriple(M->getTargetTriple());
  std::string CPUStr, FeaturesStr;
  TargetMachine *Machine = nullptr;
  const TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags();

  if (ModuleTriple.getArch()) {
    CPUStr = codegen::getCPUStr();
    FeaturesStr = codegen::getFeaturesStr();
    Machine = GetTargetMachine(ModuleTriple, CPUStr, FeaturesStr, Options);
  } else if (ModuleTriple.getArchName() != "unknown" &&
             ModuleTriple.getArchName() != "") {
    errs() << argv[0] << ": unrecognized architecture '"
           << ModuleTriple.getArchName() << "' provided.\n";
    return 1;
  }

  std::unique_ptr<TargetMachine> TM(Machine);

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(CPUStr, FeaturesStr, *M);

  // If the output is set to be emitted to standard out, and standard out is a
  // console, print out a warning message and refuse to do it.  We don't
  // impress anyone by spewing tons of binary goo to a terminal.
  if (!Force && !NoOutput && !AnalyzeOnly && !OutputAssembly)
    if (CheckBitcodeOutputToConsole(Out->os()))
      NoOutput = true;

  if (OutputThinLTOBC)
    M->addModuleFlag(Module::Error, "EnableSplitLTOUnit", SplitLTOUnit);

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(ModuleTriple);

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  else {
    // Disable individual builtin functions in TargetLibraryInfo.
    LibFunc F;
    for (auto &FuncName : DisableBuiltins)
      if (TLII.getLibFunc(FuncName, F))
        TLII.setUnavailable(F);
      else {
        errs() << argv[0] << ": cannot disable nonexistent builtin function "
               << FuncName << '\n';
        return 1;
      }
  }

  // If `-passes=` is specified, use NPM.
  // If `-enable-new-pm` is specified and there are no codegen passes, use NPM.
  // e.g. `-enable-new-pm -sroa` will use NPM.
  // but `-enable-new-pm -codegenprepare` will still revert to legacy PM.
  if ((EnableNewPassManager && !CodegenPassSpecifiedInPassList()) ||
      PassPipeline.getNumOccurrences() > 0) {
    if (PassPipeline.getNumOccurrences() > 0 && PassList.size() > 0) {
      errs()
          << "Cannot specify passes via both -foo-pass and --passes=foo-pass";
      return 1;
    }
    SmallVector<StringRef, 4> Passes;
    for (const auto &P : PassList) {
      Passes.push_back(P->getPassArgument());
    }
    if (OptLevelO0)
      Passes.push_back("default<O0>");
    if (OptLevelO1)
      Passes.push_back("default<O1>");
    if (OptLevelO2)
      Passes.push_back("default<O2>");
    if (OptLevelO3)
      Passes.push_back("default<O3>");
    if (OptLevelOs)
      Passes.push_back("default<Os>");
    if (OptLevelOz)
      Passes.push_back("default<Oz>");
    OutputKind OK = OK_NoOutput;
    if (!NoOutput)
      OK = OutputAssembly
               ? OK_OutputAssembly
               : (OutputThinLTOBC ? OK_OutputThinLTOBitcode : OK_OutputBitcode);

    VerifierKind VK = VK_VerifyInAndOut;
    if (NoVerify)
      VK = VK_NoVerifier;
    else if (VerifyEach)
      VK = VK_VerifyEachPass;

    // The user has asked to use the new pass manager and provided a pipeline
    // string. Hand off the rest of the functionality to the new code for that
    // layer.
    return runPassPipeline(argv[0], *M, TM.get(), &TLII, Out.get(),
                           ThinLinkOut.get(), RemarksFile.get(), PassPipeline,
                           Passes, OK, VK, PreserveAssemblyUseListOrder,
                           PreserveBitcodeUseListOrder, EmitSummaryIndex,
                           EmitModuleHash, EnableDebugify, Coroutines)
               ? 0
               : 1;
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build. If the -debugify-each option is set, wrap each pass with
  // the (-check)-debugify passes.
  DebugifyCustomPassManager Passes;
  if (DebugifyEach)
    Passes.enableDebugifyEach();

  bool AddOneTimeDebugifyPasses = EnableDebugify && !DebugifyEach;

  Passes.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis()
                                                     : TargetIRAnalysis()));

  if (AddOneTimeDebugifyPasses)
    Passes.add(createDebugifyModulePass());

  std::unique_ptr<legacy::FunctionPassManager> FPasses;
  if (OptLevelO0 || OptLevelO1 || OptLevelO2 || OptLevelOs || OptLevelOz ||
      OptLevelO3) {
    FPasses.reset(new legacy::FunctionPassManager(M.get()));
    FPasses->add(createTargetTransformInfoWrapperPass(
        TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));
  }

  if (PrintBreakpoints) {
    // Default to standard output.
    if (!Out) {
      if (OutputFilename.empty())
        OutputFilename = "-";

      std::error_code EC;
      Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                              sys::fs::OF_None);
      if (EC) {
        errs() << EC.message() << '\n';
        return 1;
      }
    }
    Passes.add(createBreakpointPrinter(Out->os()));
    NoOutput = true;
  }

  if (TM) {
    // FIXME: We should dyn_cast this when supported.
    auto &LTM = static_cast<LLVMTargetMachine &>(*TM);
    Pass *TPC = LTM.createPassConfig(Passes);
    Passes.add(TPC);
  }

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < PassList.size(); ++i) {
    if (StandardLinkOpts &&
        StandardLinkOpts.getPosition() < PassList.getPosition(i)) {
      AddStandardLinkPasses(Passes);
      StandardLinkOpts = false;
    }

    if (OptLevelO0 && OptLevelO0.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 0, 0);
      OptLevelO0 = false;
    }

    if (OptLevelO1 && OptLevelO1.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 1, 0);
      OptLevelO1 = false;
    }

    if (OptLevelO2 && OptLevelO2.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 0);
      OptLevelO2 = false;
    }

    if (OptLevelOs && OptLevelOs.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 1);
      OptLevelOs = false;
    }

    if (OptLevelOz && OptLevelOz.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 2);
      OptLevelOz = false;
    }

    if (OptLevelO3 && OptLevelO3.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, TM.get(), 3, 0);
      OptLevelO3 = false;
    }

    const PassInfo *PassInf = PassList[i];
    Pass *P = nullptr;
    if (PassInf->getNormalCtor())
      P = PassInf->getNormalCtor()();
    else
      errs() << argv[0] << ": cannot create pass: "
             << PassInf->getPassName() << "\n";
    if (P) {
      PassKind Kind = P->getPassKind();
      addPass(Passes, P);

      if (AnalyzeOnly) {
        switch (Kind) {
        case PT_Region:
          Passes.add(createRegionPassPrinter(PassInf, Out->os()));
          break;
        case PT_Loop:
          Passes.add(createLoopPassPrinter(PassInf, Out->os()));
          break;
        case PT_Function:
          Passes.add(createFunctionPassPrinter(PassInf, Out->os()));
          break;
        case PT_CallGraphSCC:
          Passes.add(createCallGraphPassPrinter(PassInf, Out->os()));
          break;
        default:
          Passes.add(createModulePassPrinter(PassInf, Out->os()));
          break;
        }
      }
    }

    if (PrintEachXForm)
      Passes.add(
          createPrintModulePass(errs(), "", PreserveAssemblyUseListOrder));
  }

  if (StandardLinkOpts) {
    AddStandardLinkPasses(Passes);
    StandardLinkOpts = false;
  }

  if (OptLevelO0)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 0, 0);

  if (OptLevelO1)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 1, 0);

  if (OptLevelO2)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 0);

  if (OptLevelOs)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 1);

  if (OptLevelOz)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 2, 2);

  if (OptLevelO3)
    AddOptimizationPasses(Passes, *FPasses, TM.get(), 3, 0);

  if (FPasses) {
    FPasses->doInitialization();
    for (Function &F : *M)
      FPasses->run(F);
    FPasses->doFinalization();
  }

  // Check that the module is well formed on completion of optimization
  if (!NoVerify && !VerifyEach)
    Passes.add(createVerifierPass());

  if (AddOneTimeDebugifyPasses)
    Passes.add(createCheckDebugifyModulePass(false));

  // In run twice mode, we want to make sure the output is bit-by-bit
  // equivalent if we run the pass manager again, so setup two buffers and
  // a stream to write to them. Note that llc does something similar and it
  // may be worth to abstract this out in the future.
  SmallVector<char, 0> Buffer;
  SmallVector<char, 0> FirstRunBuffer;
  std::unique_ptr<raw_svector_ostream> BOS;
  raw_ostream *OS = nullptr;

  const bool ShouldEmitOutput = !NoOutput && !AnalyzeOnly;

  // Write bitcode or assembly to the output as the last step...
  if (ShouldEmitOutput || RunTwice) {
    assert(Out);
    OS = &Out->os();
    if (RunTwice) {
      BOS = std::make_unique<raw_svector_ostream>(Buffer);
      OS = BOS.get();
    }
    if (OutputAssembly) {
      if (EmitSummaryIndex)
        report_fatal_error("Text output is incompatible with -module-summary");
      if (EmitModuleHash)
        report_fatal_error("Text output is incompatible with -module-hash");
      Passes.add(createPrintModulePass(*OS, "", PreserveAssemblyUseListOrder));
    } else if (OutputThinLTOBC)
      Passes.add(createWriteThinLTOBitcodePass(
          *OS, ThinLinkOut ? &ThinLinkOut->os() : nullptr));
    else
      Passes.add(createBitcodeWriterPass(*OS, PreserveBitcodeUseListOrder,
                                         EmitSummaryIndex, EmitModuleHash));
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  if (!RunTwice) {
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
  } else {
    // If requested, run all passes twice with the same pass manager to catch
    // bugs caused by persistent state in the passes.
    std::unique_ptr<Module> M2(CloneModule(*M));
    // Run all passes on the original module first, so the second run processes
    // the clone to catch CloneModule bugs.
    Passes.run(*M);
    FirstRunBuffer = Buffer;
    Buffer.clear();

    Passes.run(*M2);

    // Compare the two outputs and make sure they're the same
    assert(Out);
    if (Buffer.size() != FirstRunBuffer.size() ||
        (memcmp(Buffer.data(), FirstRunBuffer.data(), Buffer.size()) != 0)) {
      errs()
          << "Running the pass manager twice changed the output.\n"
             "Writing the result of the second run to the specified output.\n"
             "To generate the one-run comparison binary, just run without\n"
             "the compile-twice option\n";
      if (ShouldEmitOutput) {
        Out->os() << BOS->str();
        Out->keep();
      }
      if (RemarksFile)
        RemarksFile->keep();
      return 1;
    }
    if (ShouldEmitOutput)
      Out->os() << BOS->str();
  }

  if (DebugifyEach && !DebugifyExport.empty())
    exportDebugifyStats(DebugifyExport, Passes.getDebugifyStatsMap());

  // Declare success.
  if (!NoOutput || PrintBreakpoints)
    Out->keep();

  if (RemarksFile)
    RemarksFile->keep();

  if (ThinLinkOut)
    ThinLinkOut->keep();

  return 0;
}
