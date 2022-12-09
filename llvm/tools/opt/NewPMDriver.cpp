//===- NewPMDriver.cpp - Driver for opt with new PM -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file is just a split of the code that logically belongs in opt.cpp but
/// that includes the new pass manager headers.
///
//===----------------------------------------------------------------------===//

#include "NewPMDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/ThinLTOBitcodeWriter.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/Debugify.h"

using namespace llvm;
using namespace opt_tool;

namespace llvm {
cl::opt<bool> DebugifyEach(
    "debugify-each",
    cl::desc("Start each pass with debugify and end it with check-debugify"));

cl::opt<std::string>
    DebugifyExport("debugify-export",
                   cl::desc("Export per-pass debugify statistics to this file"),
                   cl::value_desc("filename"));

cl::opt<bool> VerifyEachDebugInfoPreserve(
    "verify-each-debuginfo-preserve",
    cl::desc("Start each pass with collecting and end it with checking of "
             "debug info preservation."));

cl::opt<std::string>
    VerifyDIPreserveExport("verify-di-preserve-export",
                   cl::desc("Export debug info preservation failures into "
                            "specified (JSON) file (should be abs path as we use"
                            " append mode to insert new JSON objects)"),
                   cl::value_desc("filename"), cl::init(""));

} // namespace llvm

enum class DebugLogging { None, Normal, Verbose, Quiet };

static cl::opt<DebugLogging> DebugPM(
    "debug-pass-manager", cl::Hidden, cl::ValueOptional,
    cl::desc("Print pass management debugging information"),
    cl::init(DebugLogging::None),
    cl::values(
        clEnumValN(DebugLogging::Normal, "", ""),
        clEnumValN(DebugLogging::Quiet, "quiet",
                   "Skip printing info about analyses"),
        clEnumValN(
            DebugLogging::Verbose, "verbose",
            "Print extra information about adaptors and pass managers")));

// This flag specifies a textual description of the alias analysis pipeline to
// use when querying for aliasing information. It only works in concert with
// the "passes" flag above.
static cl::opt<std::string>
    AAPipeline("aa-pipeline",
               cl::desc("A textual description of the alias analysis "
                        "pipeline for handling managed aliasing queries"),
               cl::Hidden, cl::init("default"));

/// {{@ These options accept textual pipeline descriptions which will be
/// inserted into default pipelines at the respective extension points
static cl::opt<std::string> PeepholeEPPipeline(
    "passes-ep-peephole",
    cl::desc("A textual description of the function pass pipeline inserted at "
             "the Peephole extension points into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> LateLoopOptimizationsEPPipeline(
    "passes-ep-late-loop-optimizations",
    cl::desc(
        "A textual description of the loop pass pipeline inserted at "
        "the LateLoopOptimizations extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> LoopOptimizerEndEPPipeline(
    "passes-ep-loop-optimizer-end",
    cl::desc("A textual description of the loop pass pipeline inserted at "
             "the LoopOptimizerEnd extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> ScalarOptimizerLateEPPipeline(
    "passes-ep-scalar-optimizer-late",
    cl::desc("A textual description of the function pass pipeline inserted at "
             "the ScalarOptimizerLate extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> CGSCCOptimizerLateEPPipeline(
    "passes-ep-cgscc-optimizer-late",
    cl::desc("A textual description of the cgscc pass pipeline inserted at "
             "the CGSCCOptimizerLate extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> VectorizerStartEPPipeline(
    "passes-ep-vectorizer-start",
    cl::desc("A textual description of the function pass pipeline inserted at "
             "the VectorizerStart extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> PipelineStartEPPipeline(
    "passes-ep-pipeline-start",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the PipelineStart extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> PipelineEarlySimplificationEPPipeline(
    "passes-ep-pipeline-early-simplification",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the EarlySimplification extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> OptimizerEarlyEPPipeline(
    "passes-ep-optimizer-early",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the OptimizerEarly extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> OptimizerLastEPPipeline(
    "passes-ep-optimizer-last",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the OptimizerLast extension point into default pipelines"),
    cl::Hidden);
static cl::opt<std::string> FullLinkTimeOptimizationEarlyEPPipeline(
    "passes-ep-full-link-time-optimization-early",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the FullLinkTimeOptimizationEarly extension point into default "
             "pipelines"),
    cl::Hidden);
static cl::opt<std::string> FullLinkTimeOptimizationLastEPPipeline(
    "passes-ep-full-link-time-optimization-last",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the FullLinkTimeOptimizationLast extension point into default "
             "pipelines"),
    cl::Hidden);

static cl::opt<bool> DisablePipelineVerification(
    "disable-pipeline-verification",
    cl::desc("Only has an effect when specified with -print-pipeline-passes. "
             "Disables verifying that the textual pipeline generated by "
             "-print-pipeline-passes can be used to create a pipeline."),
    cl::Hidden);

// Individual pipeline tuning options.
extern cl::opt<bool> DisableLoopUnrolling;

namespace llvm {
extern cl::opt<PGOKind> PGOKindFlag;
extern cl::opt<std::string> ProfileFile;
extern cl::opt<CSPGOKind> CSPGOKindFlag;
extern cl::opt<std::string> CSProfileGenFile;
extern cl::opt<bool> DisableBasicAA;
extern cl::opt<bool> PrintPipelinePasses;
} // namespace llvm

static cl::opt<std::string>
    ProfileRemappingFile("profile-remapping-file",
                         cl::desc("Path to the profile remapping file."),
                         cl::Hidden);
static cl::opt<bool> DebugInfoForProfiling(
    "new-pm-debug-info-for-profiling", cl::init(false), cl::Hidden,
    cl::desc("Emit special debug info to enable PGO profile generation."));
static cl::opt<bool> PseudoProbeForProfiling(
    "new-pm-pseudo-probe-for-profiling", cl::init(false), cl::Hidden,
    cl::desc("Emit pseudo probes to enable PGO profile generation."));
/// @}}

template <typename PassManagerT>
bool tryParsePipelineText(PassBuilder &PB,
                          const cl::opt<std::string> &PipelineOpt) {
  if (PipelineOpt.empty())
    return false;

  // Verify the pipeline is parseable:
  PassManagerT PM;
  if (auto Err = PB.parsePassPipeline(PM, PipelineOpt)) {
    errs() << "Could not parse -" << PipelineOpt.ArgStr
           << " pipeline: " << toString(std::move(Err))
           << "... I'm going to ignore it.\n";
    return false;
  }
  return true;
}

/// If one of the EPPipeline command line options was given, register callbacks
/// for parsing and inserting the given pipeline
static void registerEPCallbacks(PassBuilder &PB) {
  if (tryParsePipelineText<FunctionPassManager>(PB, PeepholeEPPipeline))
    PB.registerPeepholeEPCallback(
        [&PB](FunctionPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse PeepholeEP pipeline: ");
          Err(PB.parsePassPipeline(PM, PeepholeEPPipeline));
        });
  if (tryParsePipelineText<LoopPassManager>(PB,
                                            LateLoopOptimizationsEPPipeline))
    PB.registerLateLoopOptimizationsEPCallback(
        [&PB](LoopPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse LateLoopOptimizationsEP pipeline: ");
          Err(PB.parsePassPipeline(PM, LateLoopOptimizationsEPPipeline));
        });
  if (tryParsePipelineText<LoopPassManager>(PB, LoopOptimizerEndEPPipeline))
    PB.registerLoopOptimizerEndEPCallback(
        [&PB](LoopPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse LoopOptimizerEndEP pipeline: ");
          Err(PB.parsePassPipeline(PM, LoopOptimizerEndEPPipeline));
        });
  if (tryParsePipelineText<FunctionPassManager>(PB,
                                                ScalarOptimizerLateEPPipeline))
    PB.registerScalarOptimizerLateEPCallback(
        [&PB](FunctionPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse ScalarOptimizerLateEP pipeline: ");
          Err(PB.parsePassPipeline(PM, ScalarOptimizerLateEPPipeline));
        });
  if (tryParsePipelineText<CGSCCPassManager>(PB, CGSCCOptimizerLateEPPipeline))
    PB.registerCGSCCOptimizerLateEPCallback(
        [&PB](CGSCCPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse CGSCCOptimizerLateEP pipeline: ");
          Err(PB.parsePassPipeline(PM, CGSCCOptimizerLateEPPipeline));
        });
  if (tryParsePipelineText<FunctionPassManager>(PB, VectorizerStartEPPipeline))
    PB.registerVectorizerStartEPCallback(
        [&PB](FunctionPassManager &PM, OptimizationLevel Level) {
          ExitOnError Err("Unable to parse VectorizerStartEP pipeline: ");
          Err(PB.parsePassPipeline(PM, VectorizerStartEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(PB, PipelineStartEPPipeline))
    PB.registerPipelineStartEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err("Unable to parse PipelineStartEP pipeline: ");
          Err(PB.parsePassPipeline(PM, PipelineStartEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(
          PB, PipelineEarlySimplificationEPPipeline))
    PB.registerPipelineEarlySimplificationEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err("Unable to parse EarlySimplification pipeline: ");
          Err(PB.parsePassPipeline(PM, PipelineEarlySimplificationEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(PB, OptimizerEarlyEPPipeline))
    PB.registerOptimizerEarlyEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err("Unable to parse OptimizerEarlyEP pipeline: ");
          Err(PB.parsePassPipeline(PM, OptimizerEarlyEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(PB, OptimizerLastEPPipeline))
    PB.registerOptimizerLastEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err("Unable to parse OptimizerLastEP pipeline: ");
          Err(PB.parsePassPipeline(PM, OptimizerLastEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(
          PB, FullLinkTimeOptimizationEarlyEPPipeline))
    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err(
              "Unable to parse FullLinkTimeOptimizationEarlyEP pipeline: ");
          Err(PB.parsePassPipeline(PM,
                                   FullLinkTimeOptimizationEarlyEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(
          PB, FullLinkTimeOptimizationLastEPPipeline))
    PB.registerFullLinkTimeOptimizationLastEPCallback(
        [&PB](ModulePassManager &PM, OptimizationLevel) {
          ExitOnError Err(
              "Unable to parse FullLinkTimeOptimizationLastEP pipeline: ");
          Err(PB.parsePassPipeline(PM, FullLinkTimeOptimizationLastEPPipeline));
        });
}

#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"

bool llvm::runPassPipeline(StringRef Arg0, Module &M, TargetMachine *TM,
                           TargetLibraryInfoImpl *TLII, ToolOutputFile *Out,
                           ToolOutputFile *ThinLTOLinkOut,
                           ToolOutputFile *OptRemarkFile,
                           StringRef PassPipeline, ArrayRef<StringRef> Passes,
                           ArrayRef<PassPlugin> PassPlugins,
                           OutputKind OK, VerifierKind VK,
                           bool ShouldPreserveAssemblyUseListOrder,
                           bool ShouldPreserveBitcodeUseListOrder,
                           bool EmitSummaryIndex, bool EmitModuleHash,
                           bool EnableDebugify, bool VerifyDIPreserve) {
  bool VerifyEachPass = VK == VK_VerifyEachPass;

  Optional<PGOOptions> P;
  switch (PGOKindFlag) {
  case InstrGen:
    P = PGOOptions(ProfileFile, "", "", PGOOptions::IRInstr);
    break;
  case InstrUse:
    P = PGOOptions(ProfileFile, "", ProfileRemappingFile, PGOOptions::IRUse);
    break;
  case SampleUse:
    P = PGOOptions(ProfileFile, "", ProfileRemappingFile,
                   PGOOptions::SampleUse);
    break;
  case NoPGO:
    if (DebugInfoForProfiling || PseudoProbeForProfiling)
      P = PGOOptions("", "", "", PGOOptions::NoAction, PGOOptions::NoCSAction,
                     DebugInfoForProfiling, PseudoProbeForProfiling);
    else
      P = None;
  }
  if (CSPGOKindFlag != NoCSPGO) {
    if (P && (P->Action == PGOOptions::IRInstr ||
              P->Action == PGOOptions::SampleUse))
      errs() << "CSPGOKind cannot be used with IRInstr or SampleUse";
    if (CSPGOKindFlag == CSInstrGen) {
      if (CSProfileGenFile.empty())
        errs() << "CSInstrGen needs to specify CSProfileGenFile";
      if (P) {
        P->CSAction = PGOOptions::CSIRInstr;
        P->CSProfileGenFile = CSProfileGenFile;
      } else
        P = PGOOptions("", CSProfileGenFile, ProfileRemappingFile,
                       PGOOptions::NoAction, PGOOptions::CSIRInstr);
    } else /* CSPGOKindFlag == CSInstrUse */ {
      if (!P)
        errs() << "CSInstrUse needs to be together with InstrUse";
      P->CSAction = PGOOptions::CSIRUse;
    }
  }
  if (TM)
    TM->setPGOOption(P);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  PrintPassOptions PrintPassOpts;
  PrintPassOpts.Verbose = DebugPM == DebugLogging::Verbose;
  PrintPassOpts.SkipAnalyses = DebugPM == DebugLogging::Quiet;
  StandardInstrumentations SI(M.getContext(), DebugPM != DebugLogging::None,
                              VerifyEachPass, PrintPassOpts);
  SI.registerCallbacks(PIC, &FAM);
  DebugifyEachInstrumentation Debugify;
  DebugifyStatsMap DIStatsMap;
  DebugInfoPerPass DebugInfoBeforePass;
  if (DebugifyEach) {
    Debugify.setDIStatsMap(DIStatsMap);
    Debugify.setDebugifyMode(DebugifyMode::SyntheticDebugInfo);
    Debugify.registerCallbacks(PIC);
  } else if (VerifyEachDebugInfoPreserve) {
    Debugify.setDebugInfoBeforePass(DebugInfoBeforePass);
    Debugify.setDebugifyMode(DebugifyMode::OriginalDebugInfo);
    Debugify.setOrigDIVerifyBugsReportFilePath(
      VerifyDIPreserveExport);
    Debugify.registerCallbacks(PIC);
  }

  PipelineTuningOptions PTO;
  // LoopUnrolling defaults on to true and DisableLoopUnrolling is initialized
  // to false above so we shouldn't necessarily need to check whether or not the
  // option has been enabled.
  PTO.LoopUnrolling = !DisableLoopUnrolling;
  PassBuilder PB(TM, PTO, P, &PIC);
  registerEPCallbacks(PB);

  // For any loaded plugins, let them register pass builder callbacks.
  for (auto &PassPlugin : PassPlugins)
    PassPlugin.registerPassBuilderCallbacks(PB);

#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"

  // Specially handle the alias analysis manager so that we can register
  // a custom pipeline of AA passes with it.
  AAManager AA;
  if (auto Err = PB.parseAAPipeline(AA, AAPipeline)) {
    errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
    return false;
  }

  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return std::move(AA); });
  // Register our TargetLibraryInfoImpl.
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());
  if (EnableDebugify)
    MPM.addPass(NewPMDebugifyPass());
  if (VerifyDIPreserve)
    MPM.addPass(NewPMDebugifyPass(DebugifyMode::OriginalDebugInfo, "",
                                  &DebugInfoBeforePass));

  // Add passes according to the -passes options.
  if (!PassPipeline.empty()) {
    assert(Passes.empty() &&
           "PassPipeline and Passes should not both contain passes");
    if (auto Err = PB.parsePassPipeline(MPM, PassPipeline)) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }
  // Add passes specified using the legacy PM syntax (i.e. not using
  // -passes). This should be removed later when such support has been
  // deprecated, i.e. when all lit tests running opt (and not using
  // -enable-new-pm=0) have been updated to use -passes.
  for (auto PassName : Passes) {
    if (auto Err = PB.parsePassPipeline(MPM, PassName)) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }

  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());
  if (EnableDebugify)
    MPM.addPass(NewPMCheckDebugifyPass(false, "", &DIStatsMap));
  if (VerifyDIPreserve)
    MPM.addPass(NewPMCheckDebugifyPass(
        false, "", nullptr, DebugifyMode::OriginalDebugInfo,
        &DebugInfoBeforePass, VerifyDIPreserveExport));

  // Add any relevant output pass at the end of the pipeline.
  switch (OK) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputAssembly:
    MPM.addPass(PrintModulePass(
        Out->os(), "", ShouldPreserveAssemblyUseListOrder, EmitSummaryIndex));
    break;
  case OK_OutputBitcode:
    MPM.addPass(BitcodeWriterPass(Out->os(), ShouldPreserveBitcodeUseListOrder,
                                  EmitSummaryIndex, EmitModuleHash));
    break;
  case OK_OutputThinLTOBitcode:
    MPM.addPass(ThinLTOBitcodeWriterPass(
        Out->os(), ThinLTOLinkOut ? &ThinLTOLinkOut->os() : nullptr));
    break;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Print a textual, '-passes=' compatible, representation of pipeline if
  // requested.
  if (PrintPipelinePasses) {
    std::string Pipeline;
    raw_string_ostream SOS(Pipeline);
    MPM.printPipeline(SOS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    outs() << Pipeline;
    outs() << "\n";

    if (!DisablePipelineVerification) {
      // Check that we can parse the returned pipeline string as an actual
      // pipeline.
      ModulePassManager TempPM;
      if (auto Err = PB.parsePassPipeline(TempPM, Pipeline)) {
        errs() << "Could not parse dumped pass pipeline: "
               << toString(std::move(Err)) << "\n";
        return false;
      }
    }

    return true;
  }

  // Now that we have all of the passes ready, run them.
  MPM.run(M, MAM);

  // Declare success.
  if (OK != OK_NoOutput) {
    Out->keep();
    if (OK == OK_OutputThinLTOBitcode && ThinLTOLinkOut)
      ThinLTOLinkOut->keep();
  }

  if (OptRemarkFile)
    OptRemarkFile->keep();

  if (DebugifyEach && !DebugifyExport.empty())
    exportDebugifyStats(DebugifyExport, Debugify.getDebugifyStatsMap());

  return true;
}

void llvm::printPasses(raw_ostream &OS) {
  PassBuilder PB;
  PB.printPassNames(OS);
}
