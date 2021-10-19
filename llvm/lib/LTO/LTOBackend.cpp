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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"

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

namespace llvm {
extern cl::opt<bool> NoPGOWarnMismatch;
}

[[noreturn]] static void reportOpenError(StringRef Path, Twine Msg) {
  errs() << "failed to open " << Path << ": " << Msg << '\n';
  errs().flush();
  exit(1);
}

Error Config::addSaveTemps(std::string OutputFileName,
                           bool UseInputModulePath) {
  ShouldDiscardValueNames = false;

  std::error_code EC;
  ResolutionFile =
      std::make_unique<raw_fd_ostream>(OutputFileName + "resolution.txt", EC,
                                       sys::fs::OpenFlags::OF_TextWithCRLF);
  if (EC) {
    ResolutionFile.reset();
    return errorCodeToError(EC);
  }

  auto setHook = [&](std::string PathSuffix, ModuleHookFn &Hook) {
    // Keep track of the hook provided by the linker, which also needs to run.
    ModuleHookFn LinkerHook = Hook;
    Hook = [=](unsigned Task, const Module &M) {
      // If the linker's hook returned false, we need to pass that result
      // through.
      if (LinkerHook && !LinkerHook(Task, M))
        return false;

      std::string PathPrefix;
      // If this is the combined module (not a ThinLTO backend compile) or the
      // user hasn't requested using the input module's path, emit to a file
      // named from the provided OutputFileName with the Task ID appended.
      if (M.getModuleIdentifier() == "ld-temp.o" || !UseInputModulePath) {
        PathPrefix = OutputFileName;
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

  setHook("0.preopt", PreOptModuleHook);
  setHook("1.promote", PostPromoteModuleHook);
  setHook("2.internalize", PostInternalizeModuleHook);
  setHook("3.import", PostImportModuleHook);
  setHook("4.opt", PostOptModuleHook);
  setHook("5.precodegen", PreCodeGenModuleHook);

  CombinedIndexHook =
      [=](const ModuleSummaryIndex &Index,
          const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols) {
        std::string Path = OutputFileName + "index.bc";
        std::error_code EC;
        raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::OF_None);
        // Because -save-temps is a debugging feature, we report the error
        // directly and exit.
        if (EC)
          reportOpenError(Path, EC.message());
        WriteIndexToFile(Index, OS);

        Path = OutputFileName + "index.dot";
        raw_fd_ostream OSDot(Path, EC, sys::fs::OpenFlags::OF_None);
        if (EC)
          reportOpenError(Path, EC.message());
        Index.exportToDot(OSDot, GUIDPreservedSymbols);
        return true;
      };

  return Error::success();
}

#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"

static void RegisterPassPlugins(ArrayRef<std::string> PassPlugins,
                                PassBuilder &PB) {
#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"

  // Load requested pass plugins and let them register pass builder callbacks
  for (auto &PluginFN : PassPlugins) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (!PassPlugin) {
      errs() << "Failed to load passes from '" << PluginFN
             << "'. Request ignored.\n";
      continue;
    }

    PassPlugin->registerPassBuilderCallbacks(PB);
  }
}

static std::unique_ptr<TargetMachine>
createTargetMachine(const Config &Conf, const Target *TheTarget, Module &M) {
  StringRef TheTriple = M.getTargetTriple();
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple(TheTriple));
  for (const std::string &A : Conf.MAttrs)
    Features.AddFeature(A);

  Optional<Reloc::Model> RelocModel = None;
  if (Conf.RelocModel)
    RelocModel = *Conf.RelocModel;
  else if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  Optional<CodeModel::Model> CodeModel;
  if (Conf.CodeModel)
    CodeModel = *Conf.CodeModel;
  else
    CodeModel = M.getCodeModel();

  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TheTriple, Conf.CPU, Features.getString(), Conf.Options, RelocModel,
      CodeModel, Conf.CGOptLevel));
  assert(TM && "Failed to create target machine");
  return TM;
}

static void runNewPMPasses(const Config &Conf, Module &Mod, TargetMachine *TM,
                           unsigned OptLevel, bool IsThinLTO,
                           ModuleSummaryIndex *ExportSummary,
                           const ModuleSummaryIndex *ImportSummary) {
  Optional<PGOOptions> PGOOpt;
  if (!Conf.SampleProfile.empty())
    PGOOpt = PGOOptions(Conf.SampleProfile, "", Conf.ProfileRemapping,
                        PGOOptions::SampleUse, PGOOptions::NoCSAction, true);
  else if (Conf.RunCSIRInstr) {
    PGOOpt = PGOOptions("", Conf.CSIRProfile, Conf.ProfileRemapping,
                        PGOOptions::IRUse, PGOOptions::CSIRInstr,
                        Conf.AddFSDiscriminator);
  } else if (!Conf.CSIRProfile.empty()) {
    PGOOpt = PGOOptions(Conf.CSIRProfile, "", Conf.ProfileRemapping,
                        PGOOptions::IRUse, PGOOptions::CSIRUse,
                        Conf.AddFSDiscriminator);
    NoPGOWarnMismatch = !Conf.PGOWarnMismatch;
  } else if (Conf.AddFSDiscriminator) {
    PGOOpt = PGOOptions("", "", "", PGOOptions::NoAction,
                        PGOOptions::NoCSAction, true);
  }
  if (TM)
    TM->setPGOOption(PGOOpt);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(Conf.DebugPassManager);
  SI.registerCallbacks(PIC, &FAM);
  PassBuilder PB(TM, Conf.PTO, PGOOpt, &PIC);

  RegisterPassPlugins(Conf.PassPlugins, PB);

  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(Triple(TM->getTargetTriple())));
  if (Conf.Freestanding)
    TLII->disableAllFunctions();
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  AAManager AA;
  // Parse a custom AA pipeline if asked to.
  if (!Conf.AAPipeline.empty()) {
    if (auto Err = PB.parseAAPipeline(AA, Conf.AAPipeline)) {
      report_fatal_error(Twine("unable to parse AA pipeline description '") +
                         Conf.AAPipeline + "': " + toString(std::move(Err)));
    }
  } else {
    AA = PB.buildDefaultAAPipeline();
  }
  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return std::move(AA); });

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

  MPM.run(Mod, MAM);
}

static void runOldPMPasses(const Config &Conf, Module &Mod, TargetMachine *TM,
                           bool IsThinLTO, ModuleSummaryIndex *ExportSummary,
                           const ModuleSummaryIndex *ImportSummary) {
  legacy::PassManager passes;
  passes.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM->getTargetTriple()));
  if (Conf.Freestanding)
    PMB.LibraryInfo->disableAllFunctions();
  PMB.Inliner = createFunctionInliningPass();
  PMB.ExportSummary = ExportSummary;
  PMB.ImportSummary = ImportSummary;
  // Unconditionally verify input since it is not verified before this
  // point and has unknown origin.
  PMB.VerifyInput = true;
  PMB.VerifyOutput = !Conf.DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = Conf.OptLevel;
  PMB.PGOSampleUse = Conf.SampleProfile;
  PMB.EnablePGOCSInstrGen = Conf.RunCSIRInstr;
  if (!Conf.RunCSIRInstr && !Conf.CSIRProfile.empty()) {
    PMB.EnablePGOCSInstrUse = true;
    PMB.PGOInstrUse = Conf.CSIRProfile;
  }
  if (IsThinLTO)
    PMB.populateThinLTOPassManager(passes);
  else
    PMB.populateLTOPassManager(passes);
  passes.run(Mod);
}

bool lto::opt(const Config &Conf, TargetMachine *TM, unsigned Task, Module &Mod,
              bool IsThinLTO, ModuleSummaryIndex *ExportSummary,
              const ModuleSummaryIndex *ImportSummary,
              const std::vector<uint8_t> &CmdArgs) {
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
    llvm::EmbedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
                               /*EmbedBitcode*/ true, /*EmbedCmdline*/ true,
                               /*Cmdline*/ CmdArgs);
  }
  // FIXME: Plumb the combined index into the new pass manager.
  if (Conf.UseNewPM || !Conf.OptPipeline.empty()) {
    runNewPMPasses(Conf, Mod, TM, Conf.OptLevel, IsThinLTO, ExportSummary,
                   ImportSummary);
  } else {
    runOldPMPasses(Conf, Mod, TM, IsThinLTO, ExportSummary, ImportSummary);
  }
  return !Conf.PostOptModuleHook || Conf.PostOptModuleHook(Task, Mod);
}

static void codegen(const Config &Conf, TargetMachine *TM,
                    AddStreamFn AddStream, unsigned Task, Module &Mod,
                    const ModuleSummaryIndex &CombinedIndex) {
  if (Conf.PreCodeGenModuleHook && !Conf.PreCodeGenModuleHook(Task, Mod))
    return;

  if (EmbedBitcode == LTOBitcodeEmbedding::EmbedOptimized)
    llvm::EmbedBitcodeInModule(Mod, llvm::MemoryBufferRef(),
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

  auto Stream = AddStream(Task);
  legacy::PassManager CodeGenPasses;
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

static void splitCodeGen(const Config &C, TargetMachine *TM,
                         AddStreamFn AddStream,
                         unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                         const ModuleSummaryIndex &CombinedIndex) {
  ThreadPool CodegenThreadPool(
      heavyweight_hardware_concurrency(ParallelCodeGenParallelismLevel));
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  SplitModule(
      Mod, ParallelCodeGenParallelismLevel,
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
              Expected<std::unique_ptr<Module>> MOrErr = parseBitcodeFile(
                  MemoryBufferRef(StringRef(BC.data(), BC.size()), "ld-temp.o"),
                  Ctx);
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
      },
      false);

  // Because the inner lambda (which runs in a worker thread) captures our local
  // variables, we need to wait for the worker threads to terminate before we
  // can leave the function scope.
  CodegenThreadPool.wait();
}

static Expected<const Target *> initAndLookupTarget(const Config &C,
                                                    Module &Mod) {
  if (!C.OverrideTriple.empty())
    Mod.setTargetTriple(C.OverrideTriple);
  else if (Mod.getTargetTriple().empty())
    Mod.setTargetTriple(C.DefaultTriple);

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(Mod.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());
  return T;
}

Error lto::finalizeOptimizationRemarks(
    std::unique_ptr<ToolOutputFile> DiagOutputFile) {
  // Make sure we flush the diagnostic remarks file in case the linker doesn't
  // call the global destructors before exiting.
  if (!DiagOutputFile)
    return Error::success();
  DiagOutputFile->keep();
  DiagOutputFile->os().flush();
  return Error::success();
}

Error lto::backend(const Config &C, AddStreamFn AddStream,
                   unsigned ParallelCodeGenParallelismLevel, Module &Mod,
                   ModuleSummaryIndex &CombinedIndex) {
  Expected<const Target *> TOrErr = initAndLookupTarget(C, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = createTargetMachine(C, *TOrErr, Mod);

  if (!C.CodeGenOnly) {
    if (!opt(C, TM.get(), 0, Mod, /*IsThinLTO=*/false,
             /*ExportSummary=*/&CombinedIndex, /*ImportSummary=*/nullptr,
             /*CmdArgs*/ std::vector<uint8_t>()))
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
  std::vector<GlobalValue*> DeadGVs;
  for (auto &GV : Mod.global_values())
    if (GlobalValueSummary *GVS = DefinedGlobals.lookup(GV.getGUID()))
      if (!Index.isGlobalValueLive(GVS)) {
        DeadGVs.push_back(&GV);
        convertToDeclaration(GV);
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
                       const std::vector<uint8_t> &CmdArgs) {
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

  if (Conf.CodeGenOnly) {
    codegen(Conf, TM.get(), AddStream, Task, Mod, CombinedIndex);
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));
  }

  if (Conf.PreOptModuleHook && !Conf.PreOptModuleHook(Task, Mod))
    return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

  auto OptimizeAndCodegen =
      [&](Module &Mod, TargetMachine *TM,
          std::unique_ptr<ToolOutputFile> DiagnosticOutputFile) {
        if (!opt(Conf, TM, Task, Mod, /*IsThinLTO=*/true,
                 /*ExportSummary=*/nullptr, /*ImportSummary=*/&CombinedIndex,
                 CmdArgs))
          return finalizeOptimizationRemarks(std::move(DiagnosticOutputFile));

        codegen(Conf, TM, AddStream, Task, Mod, CombinedIndex);
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

  FunctionImporter Importer(CombinedIndex, ModuleLoader,
                            ClearDSOLocalOnDeclarations);
  if (Error Err = Importer.importFunctions(Mod, ImportList).takeError())
    return Err;

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
    if (GlobalList.second.SummaryList.empty())
      continue;

    auto GUID = GlobalList.first;
    for (const auto &Summary : GlobalList.second.SummaryList) {
      // Skip the summaries for the importing module. These are included to
      // e.g. record required linkage changes.
      if (Summary->modulePath() == M.getModuleIdentifier())
        continue;
      // Add an entry to provoke importing by thinBackend.
      ImportList[Summary->modulePath()].insert(GUID);
    }
  }
  return true;
}
