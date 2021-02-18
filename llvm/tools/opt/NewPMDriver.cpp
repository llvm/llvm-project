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
#include "PassPrinters.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"
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
} // namespace llvm

static cl::opt<bool>
    DebugPM("debug-pass-manager", cl::Hidden,
            cl::desc("Print pass management debugging information"));

static cl::list<std::string>
    PassPlugins("load-pass-plugin",
                cl::desc("Load passes from plugin library"));

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
static cl::opt<std::string> OptimizerLastEPPipeline(
    "passes-ep-optimizer-last",
    cl::desc("A textual description of the module pass pipeline inserted at "
             "the OptimizerLast extension point into default pipelines"),
    cl::Hidden);

// Individual pipeline tuning options.
extern cl::opt<bool> DisableLoopUnrolling;

extern cl::opt<PGOKind> PGOKindFlag;
extern cl::opt<std::string> ProfileFile;
extern cl::opt<CSPGOKind> CSPGOKindFlag;
extern cl::opt<std::string> CSProfileGenFile;
extern cl::opt<bool> DisableBasicAA;

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
static cl::opt<bool> UniqueInternalLinkageNames(
    "new-pm-unique-internal-linkage-names", cl::init(false), cl::Hidden,
    cl::desc("Uniqueify Internal Linkage Symbol Names by appending the MD5 "
             "hash of the module path."));
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
        [&PB](FunctionPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse PeepholeEP pipeline: ");
          Err(PB.parsePassPipeline(PM, PeepholeEPPipeline));
        });
  if (tryParsePipelineText<LoopPassManager>(PB,
                                            LateLoopOptimizationsEPPipeline))
    PB.registerLateLoopOptimizationsEPCallback(
        [&PB](LoopPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse LateLoopOptimizationsEP pipeline: ");
          Err(PB.parsePassPipeline(PM, LateLoopOptimizationsEPPipeline));
        });
  if (tryParsePipelineText<LoopPassManager>(PB, LoopOptimizerEndEPPipeline))
    PB.registerLoopOptimizerEndEPCallback(
        [&PB](LoopPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse LoopOptimizerEndEP pipeline: ");
          Err(PB.parsePassPipeline(PM, LoopOptimizerEndEPPipeline));
        });
  if (tryParsePipelineText<FunctionPassManager>(PB,
                                                ScalarOptimizerLateEPPipeline))
    PB.registerScalarOptimizerLateEPCallback(
        [&PB](FunctionPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse ScalarOptimizerLateEP pipeline: ");
          Err(PB.parsePassPipeline(PM, ScalarOptimizerLateEPPipeline));
        });
  if (tryParsePipelineText<CGSCCPassManager>(PB, CGSCCOptimizerLateEPPipeline))
    PB.registerCGSCCOptimizerLateEPCallback(
        [&PB](CGSCCPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse CGSCCOptimizerLateEP pipeline: ");
          Err(PB.parsePassPipeline(PM, CGSCCOptimizerLateEPPipeline));
        });
  if (tryParsePipelineText<FunctionPassManager>(PB, VectorizerStartEPPipeline))
    PB.registerVectorizerStartEPCallback(
        [&PB](FunctionPassManager &PM, PassBuilder::OptimizationLevel Level) {
          ExitOnError Err("Unable to parse VectorizerStartEP pipeline: ");
          Err(PB.parsePassPipeline(PM, VectorizerStartEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(PB, PipelineStartEPPipeline))
    PB.registerPipelineStartEPCallback(
        [&PB](ModulePassManager &PM, PassBuilder::OptimizationLevel) {
          ExitOnError Err("Unable to parse PipelineStartEP pipeline: ");
          Err(PB.parsePassPipeline(PM, PipelineStartEPPipeline));
        });
  if (tryParsePipelineText<ModulePassManager>(
          PB, PipelineEarlySimplificationEPPipeline))
    PB.registerPipelineEarlySimplificationEPCallback(
        [&PB](ModulePassManager &PM, PassBuilder::OptimizationLevel) {
          ExitOnError Err("Unable to parse EarlySimplification pipeline: ");
          Err(PB.parsePassPipeline(PM, PipelineEarlySimplificationEPPipeline));
        });
  if (tryParsePipelineText<FunctionPassManager>(PB, OptimizerLastEPPipeline))
    PB.registerOptimizerLastEPCallback(
        [&PB](ModulePassManager &PM, PassBuilder::OptimizationLevel) {
          ExitOnError Err("Unable to parse OptimizerLastEP pipeline: ");
          Err(PB.parsePassPipeline(PM, OptimizerLastEPPipeline));
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
                           OutputKind OK, VerifierKind VK,
                           bool ShouldPreserveAssemblyUseListOrder,
                           bool ShouldPreserveBitcodeUseListOrder,
                           bool EmitSummaryIndex, bool EmitModuleHash,
                           bool EnableDebugify, bool Coroutines) {
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
    if (DebugInfoForProfiling)
      P = PGOOptions("", "", "", PGOOptions::NoAction, PGOOptions::NoCSAction,
                     true);
    else if (PseudoProbeForProfiling)
      P = PGOOptions("", "", "", PGOOptions::NoAction, PGOOptions::NoCSAction,
                     false, true);
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
  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(DebugPM, VerifyEachPass);
  SI.registerCallbacks(PIC);
  DebugifyEachInstrumentation Debugify;
  if (DebugifyEach)
    Debugify.registerCallbacks(PIC);

  PipelineTuningOptions PTO;
  // LoopUnrolling defaults on to true and DisableLoopUnrolling is initialized
  // to false above so we shouldn't necessarily need to check whether or not the
  // option has been enabled.
  PTO.LoopUnrolling = !DisableLoopUnrolling;
  PTO.Coroutines = Coroutines;
  PTO.UniqueLinkageNames = UniqueInternalLinkageNames;
  PassBuilder PB(DebugPM, TM, PTO, P, &PIC);
  registerEPCallbacks(PB);

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

  // Register a callback that creates the debugify passes as needed.
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "debugify") {
          MPM.addPass(NewPMDebugifyPass());
          return true;
        } else if (Name == "check-debugify") {
          MPM.addPass(NewPMCheckDebugifyPass());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "asan-pipeline") {
          MPM.addPass(
              RequireAnalysisPass<ASanGlobalsMetadataAnalysis, Module>());
          MPM.addPass(
              createModuleToFunctionPassAdaptor(AddressSanitizerPass()));
          MPM.addPass(ModuleAddressSanitizerPass());
          return true;
        } else if (Name == "asan-function-pipeline") {
          MPM.addPass(
              RequireAnalysisPass<ASanGlobalsMetadataAnalysis, Module>());
          MPM.addPass(
              createModuleToFunctionPassAdaptor(AddressSanitizerPass()));
          return true;
        }
        return false;
      });

#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"

  // Specially handle the alias analysis manager so that we can register
  // a custom pipeline of AA passes with it.
  AAManager AA;
  if (Passes.empty()) {
    if (auto Err = PB.parseAAPipeline(AA, AAPipeline)) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }

  // For compatibility with the legacy PM AA pipeline.
  // AAResultsWrapperPass by default provides basic-aa in the legacy PM
  // unless -disable-basic-aa is specified.
  // TODO: remove this once tests implicitly requiring basic-aa use -passes= and
  // -aa-pipeline=basic-aa.
  if (!Passes.empty() && !DisableBasicAA) {
    if (auto Err = PB.parseAAPipeline(AA, "basic-aa")) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }

  // For compatibility with legacy pass manager.
  // Alias analyses are not specially specified when using the legacy PM.
  for (auto PassName : Passes) {
    if (PB.isAAPassName(PassName)) {
      if (auto Err = PB.parseAAPipeline(AA, PassName)) {
        errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
        return false;
      }
    }
  }

  LoopAnalysisManager LAM(DebugPM);
  FunctionAnalysisManager FAM(DebugPM);
  CGSCCAnalysisManager CGAM(DebugPM);
  ModuleAnalysisManager MAM(DebugPM);

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

  ModulePassManager MPM(DebugPM);
  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());
  if (EnableDebugify)
    MPM.addPass(NewPMDebugifyPass());

  if (!PassPipeline.empty()) {
    assert(Passes.empty() &&
           "PassPipeline and Passes should not both contain passes");
    if (auto Err = PB.parsePassPipeline(MPM, PassPipeline)) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }
  for (auto PassName : Passes) {
    std::string ModifiedPassName(PassName.begin(), PassName.end());
    if (PB.isAnalysisPassName(PassName))
      ModifiedPassName = "require<" + ModifiedPassName + ">";
    if (auto Err = PB.parsePassPipeline(MPM, ModifiedPassName)) {
      errs() << Arg0 << ": " << toString(std::move(Err)) << "\n";
      return false;
    }
  }

  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());
  if (EnableDebugify)
    MPM.addPass(NewPMCheckDebugifyPass());

  // Add any relevant output pass at the end of the pipeline.
  switch (OK) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputAssembly:
    MPM.addPass(
        PrintModulePass(Out->os(), "", ShouldPreserveAssemblyUseListOrder));
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
    exportDebugifyStats(DebugifyExport, Debugify.StatsMap);

  return true;
}

void llvm::printPasses(raw_ostream &OS) {
  PassBuilder PB;
  PB.printPassNames(OS);
}
