//===-LTOBackend.cpp - LLVM Link Time Optimizer Backend -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "backend" phase of LTO, i.e. it performs
// optimization and code generation on a loaded module. It is generally used
// internally by the LTO class but can also be used independently, for example
// to implement a standalone ThinLTO backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOBackend.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CGData/CodeGenData.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/SplitModuleCG.h"
#include <filesystem>
#include <optional>

using namespace llvm;
using namespace lto;

#define DEBUG_TYPE "lto-backend"

enum class LTOBitcodeEmbedding {
  DoNotEmbed = 0,
  EmbedOptimized = 1,
  EmbedPostMergePreOptimized = 2
};

static cl::opt<LTOBitcodeEmbedding> EmbedBitcode(
    "lto-embed-bitcode", cl::init(LTOBitcodeEmbedding::DoNotEmbed),
    cl::values(clEnumValN(LTOBitcodeEmbedding::DoNotEmbed, "none",
                          "Do not embed"),
               clEnumValN(LTOBitcodeEmbedding::EmbedOptimized, "optimized",
                          "Embed after all optimization passes"),
               clEnumValN(LTOBitcodeEmbedding::EmbedPostMergePreOptimized,
                          "post-merge-pre-opt",
                          "Embed post merge, but before optimizations")),
    cl::desc("Embed LLVM bitcode in object files produced by LTO"));

static cl::opt<bool> ThinLTOAssumeMerged(
    "thinlto-assume-merged", cl::init(false),
    cl::desc("Assume the input has already undergone ThinLTO function "
             "importing and the other pre-optimization pipeline changes."));

static cl::list<std::string>
    SaveModulesList("filter-save-modules", cl::value_desc("module names"),
                    cl::desc("Only save bitcode for module whose name without "
                             "path matches this for -save-temps options"),
                    cl::CommaSeparated, cl::Hidden);

static cl::opt<unsigned> ThinLTOSplitModuleSizeThreshold(
    "thinlto-split-module-size-threshold", cl::Hidden, cl::init(500),
    cl::desc("Control the amount of whether split in thinlto backend"
             "accroding to the size of a module."));

static cl::opt<float> ThinLTOSplitModuleSizeRateThreshold(
    "thinlto-split-module-size-rate-threshold", cl::Hidden, cl::init(0.5),
    cl::desc("Whether to split in thinlto backend based on the ratio of "
             "(callgraph size)/(module size)"));

static cl::opt<unsigned> ThinLTOSplitPartitions(
    "thinlto-split-partitions", cl::Hidden, cl::init(0),
    cl::desc("Control split to how many partitions in thinlto backend."));

static cl::opt<bool> ThinLTOSplit("thinlto-split", cl::init(false),
			   cl::desc("Enable split module in thinlto backend."));

namespace llvm {
extern cl::opt<bool> NoPGOWarnMismatch;
}

[[noreturn]] static void reportOpenError(StringRef Path, Twine Msg) {
  errs() << "failed to open " << Path << ": " << Msg << '\n';
  errs().flush();
  exit(1);
}

Error Config::addSaveTemps(std::string OutputFileName, bool UseInputModulePath,
                           const DenseSet<StringRef> &SaveTempsArgs) {
  ShouldDiscardValueNames = false;

  std::error_code EC;
  if (SaveTempsArgs.empty() || SaveTempsArgs.contains("resolution")) {
    ResolutionFile =
        std::make_unique<raw_fd_ostream>(OutputFileName + "resolution.txt", EC,
                                         sys::fs::OpenFlags::OF_TextWithCRLF);
    if (EC) {
      ResolutionFile.reset();
      return errorCodeToError(EC);
    }
  }

  auto setHook = [&](std::string PathSuffix, ModuleHookFn &Hook) {
    // Keep track of the hook provided by the linker, which also needs to run.
    ModuleHookFn LinkerHook = Hook;
    Hook = [=, SaveModNames = llvm::SmallVector<std::string, 1>(
                   SaveModulesList.begin(), SaveModulesList.end())](
               unsigned Task, const Module &M) {
      // If SaveModulesList is not empty, only do save-temps if the module's
      // filename (without path) matches a name in the list.
      if (!SaveModNames.empty() &&
          !llvm::is_contained(
              SaveModNames,
              std::string(llvm::sys::path::filename(M.getName()))))
        return false;

      // If the linker's hook returned false, we need to pass that result
      // through.
      if (LinkerHook && !LinkerHook(Task, M))
        return false;

      auto extract_filename = [](const std::string &path) -> std::string {
        std::filesystem::path fs_path(path);
        return fs_path.filename().string();
      };

      std::string PathPrefix;
      // If this is the combined module (not a ThinLTO backend compile) or the
      // user hasn't requested using the input module's path, emit to a file
      // named from the provided OutputFileName with the Task ID appended.
      if (M.getModuleIdentifier() == "ld-temp.o" || !UseInputModulePath) {
        PathPrefix = OutputFileName;
        if (ThinLTOSplit)
          PathPrefix += extract_filename(M.getSourceFileName()) + ".";
        if (Task != (unsigned)-1)
          PathPrefix += utostr(Task) + ".";
      } else
        PathPrefix = M.getModuleIdentifier() + ".";
      std::string Path = PathPrefix + PathSuffix + ".bc";
      std::error_code EC;
      raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::OF_None);
      // Because -save-temps is a debugging feature, we report the error
      // directly and exit.
      if (EC)
        reportOpenError(Path, EC.message());
      WriteBitcodeToFile(M, OS, /*ShouldPreserveUseListOrder=*/false);
      return true;
    };
  };

  auto SaveCombinedIndex =
      [=](const ModuleSummaryIndex &Index,
          const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols) {
        std::string Path = OutputFileName + "index.bc";
        std::error_code EC;
        raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::OF_None);
        // Because -save-temps is a debugging feature, we report the error
        // directly and exit.
        if (EC)
          reportOpenError(Path, EC.message());
        writeIndexToFile(Index, OS);

        Path = OutputFileName + "index.dot";
        raw_fd_ostream OSDot(Path, EC, sys::fs::OpenFlags::OF_Text);
        if (EC)
          reportOpenError(Path, EC.message());
        Index.exportToDot(OSDot, GUIDPreservedSymbols);
        return true;
      };

  if (SaveTempsArgs.empty()) {
    setHook("0.preopt", PreOptModuleHook);
    setHook("1.promote", PostPromoteModuleHook);
    setHook("2.internalize", PostInternalizeModuleHook);
    setHook("3.import", PostImportModuleHook);
    setHook("4.opt", PostOptModuleHook);
    setHook("5.precodegen", PreCodeGenModuleHook);
    CombinedIndexHook = SaveCombinedIndex;
  } else {
    if (SaveTempsArgs.contains("preopt"))
      setHook("0.preopt", PreOptModuleHook);
    if (SaveTempsArgs.contains("promote"))
      setHook("1.promote", PostPromoteModuleHook);
    if (SaveTempsArgs.contains("internalize"))
      setHook("2.internalize", PostInternalizeModuleHook);
    if (SaveTempsArgs.contains("import"))
      setHook("3.import", PostImportModuleHook);
    if (SaveTempsArgs.contains("opt"))
      setHook("4.opt", PostOptModuleHook);
    if (SaveTempsArgs.contains("precodegen"))
      setHook("5.precodegen", PreCodeGenModuleHook);
    if (SaveTempsArgs.contains("combinedindex"))
      CombinedIndexHook = SaveCombinedIndex;
  }

  return Error::success();
}

#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"
#undef HANDLE_EXTENSION

static void RegisterPassPlugins(const Config &Conf, PassBuilder &PB) {
#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"
#undef HANDLE_EXTENSION

  // Load requested pass plugins and let them register pass builder callbacks
  for (auto &PluginFN : Conf.PassPluginFilenames) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (!PassPlugin)
      reportFatalUsageError(PassPlugin.takeError());
    PassPlugin->registerPassBuilderCallbacks(PB);
  }

  // Register already loaded plugins
  for (auto *LoadedPlugin : Conf.LoadedPassPlugins)
    LoadedPlugin->registerPassBuilderCallbacks(PB);
}

static std::unique_ptr<TargetMachine>
createTargetMachine(const Config &Conf, const Target *TheTarget, Module &M) {
  const Triple &TheTriple = M.getTargetTriple();
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TheTriple);
  for (const std::string &A : Conf.MAttrs)
    Features.AddFeature(A);

  std::optional<Reloc::Model> RelocModel;
  if (Conf.RelocModel)
    RelocModel = *Conf.RelocModel;
  else if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel;
  if (Conf.CodeModel)
    CodeModel = *Conf.CodeModel;
  else
    CodeModel = M.getCodeModel();

  TargetOptions TargetOpts = Conf.Options;
  if (TargetOpts.MCOptions.ABIName.empty()) {
    TargetOpts.MCOptions.ABIName = M.getTargetABIFromMD();
  }

  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TheTriple, Conf.CPU, Features.getString(), TargetOpts, RelocModel,
      CodeModel, Conf.CGOptLevel));

  assert(TM && "Failed to create target machine");

  if (std::optional<uint64_t> LargeDataThreshold = M.getLargeDataThreshold())
    TM->setLargeDataThreshold(*LargeDataThreshold);

  return TM;
}

static void runNewPMPasses(const Config &Conf, Module &Mod, TargetMachine *TM,
                           unsigned OptLevel, bool IsThinLTO,
                           ModuleSummaryIndex *ExportSummary,
                           const ModuleSummaryIndex *ImportSummary,
                           const DenseSet<StringRef> &BitcodeLibFuncs) {
  std::optional<PGOOptions> PGOOpt;
  if (!Conf.SampleProfile.empty())
    PGOOpt = PGOOptions(Conf.SampleProfile, "", Conf.ProfileRemapping,
                        /*MemoryProfile=*/"", PGOOptions::SampleUse,
                        PGOOptions::NoCSAction,
                        PGOOptions::ColdFuncOpt::Default, true);
  else if (Conf.RunCSIRInstr) {
    PGOOpt = PGOOptions("", Conf.CSIRProfile, Conf.ProfileRemapping,
                        /*MemoryProfile=*/"", PGOOptions::IRUse,
                        PGOOptions::CSIRInstr, PGOOptions::ColdFuncOpt::Default,
                        Conf.AddFSDiscriminator);
  } else if (!Conf.CSIRProfile.empty()) {
    PGOOpt =
        PGOOptions(Conf.CSIRProfile, "", Conf.ProfileRemapping,
                   /*MemoryProfile=*/"", PGOOptions::IRUse, PGOOptions::CSIRUse,
                   PGOOptions::ColdFuncOpt::Default, Conf.AddFSDiscriminator);
    NoPGOWarnMismatch = !Conf.PGOWarnMismatch;
  } else if (Conf.AddFSDiscriminator) {
    PGOOpt = PGOOptions("", "", "", /*MemoryProfile=*/"", PGOOptions::NoAction,
                        PGOOptions::NoCSAction,
                        PGOOptions::ColdFuncOpt::Default, true);
  }
  TM->setPGOOption(PGOOpt);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(Mod.getContext(), Conf.DebugPassManager,
                              Conf.VerifyEach);
  SI.registerCallbacks(PIC, &MAM);
  PassBuilder PB(TM, Conf.PTO, PGOOpt, &PIC);

  RegisterPassPlugins(Conf, PB);

  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TM->getTargetTriple(), TM->Options.VecLib));
  if (Conf.Freestanding)
    TLII->disableAllFunctions();

  // Determine whether or not its safe to emit calls to each libfunc. Libfuncs
  // that might have been present in the current LTO unit, but are not, have
  // lost their only opportunity to be defined, and calls must not be emitted to
  // them.
  // FIXME: BitcodeLibFuncs isn't yet set for distributed ThinLTO.
  TargetLibraryInfo TLI(*TLII);
  for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs); I != E;
       ++I) {
    LibFunc F = static_cast<LibFunc>(I);
    if (BitcodeLibFuncs.contains(TLI.getName(F)))
      TLII->setUnavailable(F);
  }

  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  // Parse a custom AA pipeline if asked to.
  if (!Conf.AAPipeline.empty()) {
    AAManager AA;
    if (auto Err = PB.parseAAPipeline(AA, Conf.AAPipeline)) {
      report_fatal_error(Twine("unable to parse AA pipeline description '") +
                         Conf.AAPipeline + "': " + toString(std::move(Err)));
    }
    // Register the AA manager first so that our version is the one used.
    FAM.registerPass([&] { return std::move(AA); });
  }

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;

  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());

  OptimizationLevel OL;

  switch (OptLevel) {
  default:
    llvm_unreachable("Invalid optimization level");
  case 0:
    OL = OptimizationLevel::O0;
    break;
  case 1:
    OL = OptimizationLevel::O1;
    break;
  case 2:
    OL = OptimizationLevel::O2;
    break;
  case 3:
    OL = OptimizationLevel::O3;
    break;
  }

  // Parse a custom pipeline if asked to.
  if (!Conf.OptPipeline.empty()) {
    if (auto Err = PB.parsePassPipeline(MPM, Conf.OptPipeline)) {
      report_fatal_error(Twine("unable to parse pass pipeline description '") +
                         Conf.OptPipeline + "': " + toString(std::move(Err)));
    }
  } else if (IsThinLTO) {
    MPM.addPass(PB.buildThinLTODefaultPipeline(OL, ImportSummary));
  } else {
    MPM.addPass(PB.buildLTODefaultPipeline(OL, ExportSummary));
  }

  if (!Conf.DisableVerify)
    MPM.addPass(VerifierPass());

  if (PrintPipelinePasses) {
    std::string PipelineStr;
    raw_string_ostream OS(PipelineStr);
    MPM.printPipeline(OS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    outs() << "pipeline-passes: " << PipelineStr << '\n';
  }

  MPM.run(Mod, MAM);
}

static bool isEmptyModule(const Module &Mod) {
  // Module is empty if it has no functions, no globals, no inline asm and no
  // named metadata (aliases and ifuncs require functions or globals so we
  // don't need to check those explicitly).
  return Mod.empty() && Mod.global_empty() && Mod.named_metadata_empty() &&
         Mod.getModuleInlineAsm().empty();
}

bool lto::opt(const Config &Conf, TargetMachine *TM, unsigned Task, Module &Mod,
              bool IsThinLTO, ModuleSummaryIndex *ExportSummary,
              const ModuleSummaryIndex *ImportSummary,
              const std::vector<uint8_t> &CmdArgs,
              ArrayRef<StringRef> BitcodeLibFuncs) {
  llvm::TimeTraceScope timeScope("opt");
  if (EmbedBitcode == LTOBitcodeEmbedding::EmbedPostMergePreOptimized) {
    // FIXME: the motivation for capturing post-merge bitcode and command line
    // is replicating the compilation environment from bitcode, without needing
    // to understand the dependencies (the functions to be imported). This
    // assumes a clang - based invocation, case in which we have the command
    // line.
    // It's not very clear how the above motivation would map in the
    // linker-based case, so we currently don't plumb the command line args in
    // that case.
    if (CmdArgs.empty())
      LLVM_DEBUG(
          dbgs() << "Post-(Thin)LTO merge bitcode embedding was requested, but "
                    "command line arguments are not available");
    llvm::embedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
                               /*EmbedBitcode*/ true, /*EmbedCmdline*/ true,
                               /*Cmdline*/ CmdArgs);
  }
  // No need to run any opt passes if the module is empty.
  // In theory these passes should take almost no time for an empty
  // module, however, this guards against doing any unnecessary summary-based
  // analysis in the case of a ThinLTO build where this might be an empty
  // regular LTO combined module, with a large combined index from ThinLTO.
  if (!isEmptyModule(Mod)) {
    DenseSet<StringRef> BitcodeLibFuncsSet(BitcodeLibFuncs.begin(),
                                           BitcodeLibFuncs.end());
    // FIXME: Plumb the combined index into the new pass manager.
    runNewPMPasses(Conf, Mod, TM, Conf.OptLevel, IsThinLTO, ExportSummary,
                   ImportSummary, BitcodeLibFuncsSet);
  }
  return !Conf.PostOptModuleHook || Conf.PostOptModuleHook(Task, Mod);
}

static void codegen(const Config &Conf, TargetMachine *TM,
                    AddStreamFn AddStream, unsigned Task, Module &Mod,
                    const ModuleSummaryIndex &CombinedIndex) {
  llvm::TimeTraceScope timeScope("codegen");
  if (Conf.PreCodeGenModuleHook && !Conf.PreCodeGenModuleHook(Task, Mod))
    return;

  if (EmbedBitcode == LTOBitcodeEmbedding::EmbedOptimized)
    llvm::embedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
                               /*EmbedBitcode*/ true,
                               /*EmbedCmdline*/ false,
                               /*CmdArgs*/ std::vector<uint8_t>());

  std::unique_ptr<ToolOutputFile> DwoOut;
  SmallString<1024> DwoFile(Conf.SplitDwarfOutput);
  if (!Conf.DwoDir.empty()) {
    std::error_code EC;
    if (auto EC = llvm::sys::fs::create_directories(Conf.DwoDir))
      report_fatal_error(Twine("Failed to create directory ") + Conf.DwoDir +
                         ": " + EC.message());

    DwoFile = Conf.DwoDir;
    sys::path::append(DwoFile, std::to_string(Task) + ".dwo");
    TM->Options.MCOptions.SplitDwarfFile = std::string(DwoFile);
  } else
    TM->Options.MCOptions.SplitDwarfFile = Conf.SplitDwarfFile;

  if (!DwoFile.empty()) {
    std::error_code EC;
    DwoOut = std::make_unique<ToolOutputFile>(DwoFile, EC, sys::fs::OF_None);
    if (EC)
      report_fatal_error(Twine("Failed to open ") + DwoFile + ": " +
                         EC.message());
  }

  Expected<std::unique_ptr<CachedFileStream>> StreamOrErr =
      AddStream(Task, Mod.getModuleIdentifier());
  if (Error Err = StreamOrErr.takeError())
    report_fatal_error(std::move(Err));
  std::unique_ptr<CachedFileStream> &Stream = *StreamOrErr;
  TM->Options.ObjectFilenameForDebug = Stream->ObjectPathName;

  // Create the codegen pipeline in its own scope so it gets deleted before
  // Stream->commit() is called. The commit function of CacheStream deletes
  // the raw stream, which is too early as streamers (e.g. MCAsmStreamer)
  // keep the pointer and may use it until their destruction. See #138194.
  {
    legacy::PassManager CodeGenPasses;
    TargetLibraryInfoImpl TLII(Mod.getTargetTriple(), TM->Options.VecLib);
    CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
    CodeGenPasses.add(new RuntimeLibraryInfoWrapper(
        Mod.getTargetTriple(), TM->Options.ExceptionModel,
        TM->Options.FloatABIType, TM->Options.EABIVersion,
        TM->Options.MCOptions.ABIName, TM->Options.VecLib));

    // No need to make index available if the module is empty.
    // In theory these passes should not use the index for an empty
    // module, however, this guards against doing any unnecessary summary-based
    // analysis in the case of a ThinLTO build where this might be an empty
    // regular LTO combined module, with a large combined index from ThinLTO.
    if (!isEmptyModule(Mod))
      CodeGenPasses.add(
          createImmutableModuleSummaryIndexWrapperPass(&CombinedIndex));
    if (Conf.PreCodeGenPassesHook)
      Conf.PreCodeGenPassesHook(CodeGenPasses);
    if (TM->addPassesToEmitFile(CodeGenPasses, *Stream->OS,
                                DwoOut ? &DwoOut->os() : nullptr,
                                Conf.CGFileType))
      report_fatal_error("Failed to setup codegen");
    CodeGenPasses.run(Mod);

    if (DwoOut)
      DwoOut->keep();
  }

  if (Error Err = Stream->commit())
    report_fatal_error(std::move(Err));
}

static unsigned calFunctionSize(const llvm::Function &F) {
  unsigned size = 0;
  for (const auto &BB : F)
    size += std::distance(BB.begin(), BB.end());
  return size;
}

static unsigned calModuleSize(const llvm::Module &M) {
  unsigned size = 0;
  for (const auto &F : M)
    size += calFunctionSize(F);
  return size;
}

static bool canDoSplitModule(const llvm::Module &M) {
  if (calModuleSize(M) < ThinLTOSplitModuleSizeThreshold)
    return false;
  return true;
}

static bool HasLargeCG(Module &Mod, const ModuleSummaryIndex &CombinedIndex) {
  // TODO: Check whether there has large callgraphs. When multiple callgraphs
  // are split, thinlto parallel compilation can bring benefits.
  return true;
}

struct TaskIdAllocator {
  using TaskId = unsigned;

  // Use the most significant bit (MSB) as a namespace tag.
  // - Original ThinLTO backend tasks are expected to have MSB == 0.
  // - Split partitions allocated by this allocator always have MSB == 1.
  // This guarantees the two ID spaces never overlap.
  static constexpr TaskId tag() {
    return TaskId{1} << (std::numeric_limits<TaskId>::digits - 1);
  }

  // Monotonic sequence counter for split partitions (MSB must remain 0 here).
  std::atomic<TaskId> seq{0};

  // Allocate a globally unique TaskId for a split partition.
  // The returned ID is `tag() | seq`, so it lives in the MSB==1 namespace.
  TaskId alloc() {
    TaskId v = seq.fetch_add(1, std::memory_order_relaxed);

    // If the counter ever reaches the MSB, we'd overlap namespaces.
    // This indicates an overflow / too many partitions.
    if (v & tag())
      report_fatal_error("Partition TaskId overflow: seq reached the tag bit.");

    return tag() | v;
  }

  // Helper for sanity checks / debugging.
  static bool isPartition(TaskId id) { return (id & tag()) != 0; }
};

// Global allocator shared by all split partitions.
static TaskIdAllocator gSplitTaskIds;

static bool splitOptAndCodeGenThin(unsigned task, const Config &C,
                                   TargetMachine *TM, AddStreamFn AddStream,
                                   unsigned ParallelCodeGenParallelismLevel,
                                   Module &Mod,
                                   const ModuleSummaryIndex &CombinedIndex,
                                   const std::vector<uint8_t> &CmdArgs,
                                   bool DoOpt, AddStreamFn IRAddStream,
                                   ArrayRef<StringRef> &BitcodeLibFuncs) {
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  static std::mutex PrintMutex;

  SplitModuleCG SplitModuleCG(Mod, CombinedIndex, ParallelCodeGenParallelismLevel);
  ParallelCodeGenParallelismLevel = SplitModuleCG.getPartitionNum();

  std::vector<std::string> TempObjectFiles(ParallelCodeGenParallelismLevel);
  std::vector<llvm::FileRemover> TempFileRemovers(ParallelCodeGenParallelismLevel);

  const auto HandleModulePartition = [&](std::unique_ptr<Module> MPart,
                                         unsigned PartitionId) {
    unsigned CurrentThreadId, UniqueTaskId;
    {
      std::lock_guard<std::mutex> Lock(PrintMutex);
      CurrentThreadId = ThreadCount++;

      // In distributed ThinLTO, `task` may be a sentinel (e.g. -1 cast to
      // unsigned), which becomes UINT_MAX and naturally has MSB==1. Treat it
      // as "no base task id" and don't enforce the namespace check on it.
      //
      // We do not rely on the incoming `task` for partition uniqueness: split
      // partitions get a dedicated UniqueTaskId allocated below.
      if (task != std::numeric_limits<unsigned>::max()) {
        assert(!TaskIdAllocator::isPartition(task) &&
               "Original ThinLTO TaskId unexpectedly overlaps the partition "
               "namespace");
      }
      UniqueTaskId = gSplitTaskIds.alloc();
    }

    std::unique_ptr<TargetMachine> ThreadTM = createTargetMachine(C, T, *MPart);

    if (DoOpt) {
      if (!opt(C, ThreadTM.get(), UniqueTaskId, *MPart, /*IsThinLTO=*/true,
               /*ExportSummary=*/nullptr, /*ImportSummary=*/&CombinedIndex,
               CmdArgs, BitcodeLibFuncs)) {
        report_fatal_error("Failed to gen opt for split mod in thread.");
      }

      // Save the current module before the first codegen round.
      // Note that the second codegen round runs only `codegen()` without
      // running `opt()`. We're not reaching here as it's bailed out earlier
      // with `CodeGenOnly` which has been set in `SecondRoundThinBackend`.
      if (IRAddStream)
        cgdata::saveModuleForTwoRounds(*MPart, task + CurrentThreadId,
                                       IRAddStream);
    }

    auto splitStream = [&](unsigned task, const Twine &moduleName)
        -> Expected<std::unique_ptr<CachedFileStream>> {
      int FD;
      SmallString<128> TempFilename;
      if (std::error_code EC = sys::fs::createTemporaryFile(
              "thinlto-split", "o", FD, TempFilename))
        return errorCodeToError(EC);

      TempObjectFiles[PartitionId] = std::string(TempFilename.str());
      TempFileRemovers[PartitionId].setFile(TempObjectFiles[PartitionId]);

      auto OS = std::make_unique<raw_fd_ostream>(
          FD, true, /*CloseOnDestruct*/true);

      auto Stream = std::make_unique<CachedFileStream>(
          std::move(OS), std::string(TempFilename.str()));

      return std::move(Stream);
    };

    codegen(C, ThreadTM.get(), splitStream, UniqueTaskId, *MPart,
            CombinedIndex);
  };

  SplitModuleCG.SplitModule(HandleModulePartition, C);

  // Use ld.lld to combine the partitions into a object.
  if (TempObjectFiles.empty()) {
    llvm::errs() << "TempObjectFiles.empty()\n";
    return true;
  }

  auto FinalStream = AddStream(task, Mod.getModuleIdentifier());
  if (!FinalStream)
    report_fatal_error("Failed to open final output stream");

  int MergedFD;
  SmallString<128> MergedFilename;
  if (sys::fs::createTemporaryFile("thinlto-merged", "o", MergedFD,
                                   MergedFilename))
    report_fatal_error("Failed to create merged temp file.");
  llvm::FileRemover MergedFileRemover(MergedFilename);
  sys::fs::closeFile(MergedFD);

  std::vector<StringRef> Args;
  std::string LinkerPath = "";
  if (auto Path = sys::findProgramByName("ld.lld"))
    LinkerPath = *Path;
  else if (auto Path = sys::findProgramByName("ld"))
    LinkerPath = *Path;

  if (LinkerPath.empty())
    report_fatal_error("Cannot find linkeer (ld or ld.lld) to merge partitions.");

  Args.push_back(LinkerPath);
  Args.push_back("-r");
  Args.push_back("-o");
  Args.push_back(MergedFilename);

  for (const auto &File : TempObjectFiles)
    Args.push_back(File);

  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(LinkerPath, Args, /*Env=*/std::nullopt,
                                   /*Redirects=*/{}, /*SecondsToWait=*/0,
                                   /*MemoryLimit=*/0, &ErrMsg);

  if (Result != 0) {
    errs() << "Linker failed: " << ErrMsg << "\n";
    report_fatal_error("Failed to merge split objects.");
  }

  {
    std::unique_ptr<CachedFileStream> &FinalFileStream = *FinalStream;
    auto BufferOrErr = MemoryBuffer::getFile(MergedFilename);
    if (!BufferOrErr)
      report_fatal_error("Failed to read merged object.");

    FinalFileStream->OS->write(BufferOrErr.get()->getBufferStart(),
                               BufferOrErr.get()->getBufferSize());
    if (Error Err = FinalFileStream->commit()) {
      report_fatal_error(Twine("Failed to commit final file stream: ") +
                         toString(std::move(Err)));
    }
  }
  return true;
}

static void splitCodeGen(const Config &C, TargetMachine *TM,
                         AddStreamFn AddStream,
                         unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                         const ModuleSummaryIndex &CombinedIndex) {
  DefaultThreadPool CodegenThreadPool(
      heavyweight_hardware_concurrency(ParallelCodeGenParallelismLevel));
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  const auto HandleModulePartition =
      [&](std::unique_ptr<Module> MPart) {
        // We want to clone the module in a new context to multi-thread the
        // codegen. We do it by serializing partition modules to bitcode
        // (while still on the main thread, in order to avoid data races) and
        // spinning up new threads which deserialize the partitions into
        // separate contexts.
        // FIXME: Provide a more direct way to do this in LLVM.
        SmallString<0> BC;
        raw_svector_ostream BCOS(BC);
        WriteBitcodeToFile(*MPart, BCOS);

        // Enqueue the task
        CodegenThreadPool.async(
            [&](const SmallString<0> &BC, unsigned ThreadId) {
              LTOLLVMContext Ctx(C);
              Expected<std::unique_ptr<Module>> MOrErr =
                  parseBitcodeFile(MemoryBufferRef(BC.str(), "ld-temp.o"), Ctx);
              if (!MOrErr)
                report_fatal_error("Failed to read bitcode");
              std::unique_ptr<Module> MPartInCtx = std::move(MOrErr.get());

              std::unique_ptr<TargetMachine> TM =
                  createTargetMachine(C, T, *MPartInCtx);

              codegen(C, TM.get(), AddStream, ThreadId, *MPartInCtx,
                      CombinedIndex);
            },
            // Pass BC using std::move to ensure that it get moved rather than
            // copied into the thread's context.
            std::move(BC), ThreadCount++);
      };

  // Try target-specific module splitting first, then fallback to the default.
  if (!TM->splitModule(Mod, ParallelCodeGenParallelismLevel,
                       HandleModulePartition)) {
    SplitModule(Mod, ParallelCodeGenParallelismLevel, HandleModulePartition,
                false);
  }

  // Because the inner lambda (which runs in a worker thread) captures our local
  // variables, we need to wait for the worker threads to terminate before we
  // can leave the function scope.
  CodegenThreadPool.wait();
}

static Expected<const Target *> initAndLookupTarget(const Config &C,
                                                    Module &Mod) {
  if (!C.OverrideTriple.empty())
    Mod.setTargetTriple(Triple(C.OverrideTriple));
  else if (Mod.getTargetTriple().empty())
    Mod.setTargetTriple(Triple(C.DefaultTriple));

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(Mod.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());
  return T;
}

Error lto::finalizeOptimizationRemarks(LLVMRemarkFileHandle DiagOutputFile) {
  // Make sure we flush the diagnostic remarks file in case the linker doesn't
  // call the global destructors before exiting.
  if (!DiagOutputFile)
    return Error::success();
  DiagOutputFile.finalize();
  DiagOutputFile->keep();
  DiagOutputFile->os().flush();
  return Error::success();
}

Error lto::backend(const Config &C, AddStreamFn AddStream,
                   unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                   ModuleSummaryIndex &CombinedIndex,
                   ArrayRef<StringRef> BitcodeLibFuncs) {
  llvm::TimeTraceScope timeScope("LTO backend");
  Expected<const Target *> TOrErr = initAndLookupTarget(C, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = createTargetMachine(C, *TOrErr, Mod);

  LLVM_DEBUG(dbgs() << "Running regular LTO\n");
  if (!C.CodeGenOnly) {
    if (!opt(C, TM.get(), 0, Mod, /*IsThinLTO=*/false,
             /*ExportSummary=*/&CombinedIndex, /*ImportSummary=*/nullptr,
             /*CmdArgs*/ std::vector<uint8_t>(), BitcodeLibFuncs))
      return Error::success();
  }

  if (ParallelCodeGenParallelismLevel == 1) {
    codegen(C, TM.get(), AddStream, 0, Mod, CombinedIndex);
  } else {
    splitCodeGen(C, TM.get(), AddStream, ParallelCodeGenParallelismLevel, Mod,
                 CombinedIndex);
  }
  return Error::success();
}

static void dropDeadSymbols(Module &Mod, const GVSummaryMapTy &DefinedGlobals,
                            const ModuleSummaryIndex &Index) {
  llvm::TimeTraceScope timeScope("Drop dead symbols");
  std::vector<GlobalValue*> DeadGVs;

  for (auto &GV : Mod.global_values()) {
    auto GUID = GV.getGUIDIfAssigned();
    if (!GUID)
      continue;

    if (GlobalValueSummary *GVS = DefinedGlobals.lookup(*GUID))
      if (!Index.isGlobalValueLive(GVS)) {
        DeadGVs.push_back(&GV);
        convertToDeclaration(GV);
      }
  }

  // Now that all dead bodies have been dropped, delete the actual objects
  // themselves when possible.
  for (GlobalValue *GV : DeadGVs) {
    GV->removeDeadConstantUsers();
    // Might reference something defined in native object (i.e. dropped a
    // non-prevailing IR def, but we need to keep the declaration).
    if (GV->use_empty())
      GV->eraseFromParent();
  }
}

Error lto::thinBackend(const Config &Conf, unsigned Task, AddStreamFn AddStream,
                       Module &Mod, const ModuleSummaryIndex &CombinedIndex,
                       const FunctionImporter::ImportMapTy &ImportList,
                       const GVSummaryMapTy &DefinedGlobals,
                       MapVector<StringRef, BitcodeModule> *ModuleMap,
                       bool CodeGenOnly, ArrayRef<StringRef> BitcodeLibFuncs,
                       AddStreamFn IRAddStream,
                       const std::vector<uint8_t> &CmdArgs) {
  llvm::TimeTraceScope timeScope("Thin backend", Mod.getModuleIdentifier());
  Expected<const Target *> TOrErr = initAndLookupTarget(Conf, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = createTargetMachine(Conf, *TOrErr, Mod);

  // Setup optimization remarks.
  auto DiagFileOrErr = lto::setupLLVMOptimizationRemarks(
      Mod.getContext(), Conf.RemarksFilename, Conf.RemarksPasses,
      Conf.RemarksFormat, Conf.RemarksWithHotness, Conf.RemarksHotnessThreshold,
      Task);
  if (!DiagFileOrErr)
    return DiagFileOrErr.takeError();
  auto DiagnosticOutputFile = std::move(*DiagFileOrErr);

  // Set the partial sample profile ratio in the profile summary module flag of
  // the module, if applicable.
  Mod.setPartialSampleProfileRatio(CombinedIndex);

  bool ProfitableToSplit = true;
  if (ThinLTOSplit) {
    if (!canDoSplitModule(Mod) || !HasLargeCG(Mod, CombinedIndex)) {
      ProfitableToSplit = false;
      LLVM_DEBUG(dbgs() << "warning: thinlto split not enable for module: "
                        << Mod.getName());
    } else {
      LLVM_DEBUG(dbgs() << "thinlto: split codegen for module: "
                        << Mod.getName());
    }
  }

  LLVM_DEBUG(dbgs() << "Running ThinLTO\n");
  if (CodeGenOnly) {
    if (ThinLTOSplit && ProfitableToSplit)
      splitOptAndCodeGenThin(Task, Conf, TM.get(), AddStream,
                             ThinLTOSplitPartitions, Mod, CombinedIndex,
                             CmdArgs, false, IRAddStream, BitcodeLibFuncs);
    else
      // If CodeGenOnly is set, we only perform code generation and skip
      // optimization. This value may differ from Conf.CodeGenOnly.
      codegen(Conf, TM.get(), AddStream, Task, Mod, CombinedIndex);
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
  }

  if (Conf.PreOptModuleHook && !Conf.PreOptModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  auto OptimizeAndCodegen =
      [&](Module &Mod, TargetMachine *TM,
          LLVMRemarkFileHandle DiagnosticOutputFile) {
        if (ThinLTOSplit && ProfitableToSplit) {
          if (!splitOptAndCodeGenThin(
                  Task, Conf, TM, AddStream, ThinLTOSplitPartitions, Mod,
                  CombinedIndex, CmdArgs, true, IRAddStream, BitcodeLibFuncs))
            return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
        } else {
          // Perform optimization and code generation for ThinLTO.
          if (!opt(Conf, TM, Task, Mod, /*IsThinLTO=*/true,
                  /*ExportSummary=*/nullptr, /*ImportSummary=*/&CombinedIndex,
                  CmdArgs, BitcodeLibFuncs))
            return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

          // Save the current module before the first codegen round.
          // Note that the second codegen round runs only `codegen()` without
          // running `opt()`. We're not reaching here as it's bailed out earlier
          // with `CodeGenOnly` which has been set in `SecondRoundThinBackend`.
          if (IRAddStream)
            cgdata::saveModuleForTwoRounds(Mod, Task, IRAddStream);

          codegen(Conf, TM, AddStream, Task, Mod, CombinedIndex);
        }
        return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
      };

  if (ThinLTOAssumeMerged)
    return OptimizeAndCodegen(Mod, TM.get(), std::move(DiagnosticOutputFile));

  // When linking an ELF shared object, dso_local should be dropped. We
  // conservatively do this for -fpic.
  bool ClearDSOLocalOnDeclarations =
      TM->getTargetTriple().isOSBinFormatELF() &&
      TM->getRelocationModel() != Reloc::Static &&
      Mod.getPIELevel() == PIELevel::Default;
  renameModuleForThinLTO(Mod, CombinedIndex, ClearDSOLocalOnDeclarations);

  dropDeadSymbols(Mod, DefinedGlobals, CombinedIndex);

  thinLTOFinalizeInModule(Mod, DefinedGlobals, /*PropagateAttrs=*/true);

  if (Conf.PostPromoteModuleHook && !Conf.PostPromoteModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  if (!DefinedGlobals.empty())
    thinLTOInternalizeModule(Mod, DefinedGlobals);

  if (Conf.PostInternalizeModuleHook &&
      !Conf.PostInternalizeModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  auto ModuleLoader = [&](StringRef Identifier) {
    llvm::TimeTraceScope moduleLoaderScope("Module loader", Identifier);
    assert(Mod.getContext().isODRUniquingDebugTypes() &&
           "ODR Type uniquing should be enabled on the context");
    if (ModuleMap) {
      auto I = ModuleMap->find(Identifier);
      assert(I != ModuleMap->end());
      return I->second.getLazyModule(Mod.getContext(),
                                     /*ShouldLazyLoadMetadata=*/true,
                                     /*IsImporting*/ true);
    }

    ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MBOrErr =
        llvm::MemoryBuffer::getFile(Identifier);
    if (!MBOrErr)
      return Expected<std::unique_ptr<llvm::Module>>(make_error<StringError>(
          Twine("Error loading imported file ") + Identifier + " : ",
          MBOrErr.getError()));

    Expected<BitcodeModule> BMOrErr = findThinLTOModule(**MBOrErr);
    if (!BMOrErr)
      return Expected<std::unique_ptr<llvm::Module>>(make_error<StringError>(
          Twine("Error loading imported file ") + Identifier + " : " +
              toString(BMOrErr.takeError()),
          inconvertibleErrorCode()));

    Expected<std::unique_ptr<Module>> MOrErr =
        BMOrErr->getLazyModule(Mod.getContext(),
                               /*ShouldLazyLoadMetadata=*/true,
                               /*IsImporting*/ true);
    if (MOrErr)
      (*MOrErr)->setOwnedMemoryBuffer(std::move(*MBOrErr));
    return MOrErr;
  };

  {
    llvm::TimeTraceScope importScope("Import functions");
    FunctionImporter Importer(CombinedIndex, ModuleLoader,
                              ClearDSOLocalOnDeclarations);
    if (Error Err = Importer.importFunctions(Mod, ImportList).takeError())
      return Err;
  }

  // Do this after any importing so that imported code is updated.
  updatePublicTypeTestCalls(Mod, CombinedIndex.withWholeProgramVisibility());

  if (Conf.PostImportModuleHook && !Conf.PostImportModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  return OptimizeAndCodegen(Mod, TM.get(), std::move(DiagnosticOutputFile));
}

BitcodeModule *lto::findThinLTOModule(MutableArrayRef<BitcodeModule> BMs) {
  if (ThinLTOAssumeMerged && BMs.size() == 1)
    return BMs.begin();

  for (BitcodeModule &BM : BMs) {
    Expected<BitcodeLTOInfo> LTOInfo = BM.getLTOInfo();
    if (LTOInfo && LTOInfo->IsThinLTO)
      return &BM;
  }
  return nullptr;
}

Expected<BitcodeModule> lto::findThinLTOModule(MemoryBufferRef MBRef) {
  Expected<std::vector<BitcodeModule>> BMsOrErr = getBitcodeModuleList(MBRef);
  if (!BMsOrErr)
    return BMsOrErr.takeError();

  // The bitcode file may contain multiple modules, we want the one that is
  // marked as being the ThinLTO module.
  if (const BitcodeModule *Bm = lto::findThinLTOModule(*BMsOrErr))
    return *Bm;

  return make_error<StringError>("Could not find module summary",
                                 inconvertibleErrorCode());
}

bool lto::initImportList(const Module &M,
                         const ModuleSummaryIndex &CombinedIndex,
                         FunctionImporter::ImportMapTy &ImportList) {
  if (ThinLTOAssumeMerged)
    return true;
  // We can simply import the values mentioned in the combined index, since
  // we should only invoke this using the individual indexes written out
  // via a WriteIndexesThinBackend.
  for (const auto &GlobalList : CombinedIndex) {
    // Ignore entries for undefined references.
    if (GlobalList.second.getSummaryList().empty())
      continue;

    auto GUID = GlobalList.first;
    for (const auto &Summary : GlobalList.second.getSummaryList()) {
      // Skip the summaries for the importing module. These are included to
      // e.g. record required linkage changes.
      if (Summary->modulePath() == M.getModuleIdentifier())
        continue;
      // Add an entry to provoke importing by thinBackend.
      ImportList.addGUID(Summary->modulePath(), GUID, Summary->importType());
    }
  }
  return true;
}
