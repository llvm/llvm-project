//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/RunCodeGen.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/CGPassBuilderOption.h"

using namespace llvm;

static cl::opt<cl::boolOrDefault>
    ForceNewPM("force-new-pm-codegen",
               cl::desc("Whether to force the NewPM on/off. Not setting the "
                        "option will default to what the target prefers."),
               cl::init(cl::boolOrDefault::BOU_UNSET));

static Error runCodeGenPipelineLegacy(TargetMachine &TM, Module &M,
                                      raw_pwrite_stream &OS,
                                      std::unique_ptr<ToolOutputFile> &DwoOS,
                                      CodeGenFileType CGFT,
                                      const CodeGenPipelineConfig &Config) {
  legacy::PassManager CodeGenPasses;
  CodeGenPasses.add(
      createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  // Add LibraryInfo.
  TargetLibraryInfoImpl TLII(TM.getTargetTriple(), TM.Options.VecLib);
  if (Config.DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));

  const TargetOptions &Options = TM.Options;
  CodeGenPasses.add(
      new RuntimeLibraryInfoWrapper(Options.MCOptions.ABIName, Options.VecLib));

  if (Config.SummaryIndex)
    CodeGenPasses.add(
        createImmutableModuleSummaryIndexWrapperPass(Config.SummaryIndex));

  if (TM.addPassesToEmitFile(CodeGenPasses, OS, DwoOS ? &DwoOS->os() : nullptr,
                             CGFT, Config.DisableVerify))
    return createStringError("Failed to construct CodeGen pipeline");
  CodeGenPasses.run(M);

  return Error::success();
}

static Error runCodeGenPipelineNewPM(TargetMachine &TM, Module &M,
                                     raw_pwrite_stream &OS,
                                     std::unique_ptr<ToolOutputFile> &DwoOS,
                                     CodeGenFileType CGFT,
                                     const CodeGenPipelineConfig &Config) {
  ModulePassManager MPM;
  MachineFunctionAnalysisManager MFAM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  CGPassBuilderOption Opt = getCGPassBuilderOption();
  Opt.DisableVerify = Config.DisableVerify;
  MachineModuleInfo MMI(&TM);
  PassInstrumentationCallbacks PIC;
  PipelineTuningOptions PTOptions;
  TargetMachine *TMPointer = &TM;
  // Only resolve the fallback once we know we need one: under clang's IO
  // sandbox, calling vfs::getRealFileSystem() at all is fatal.
  PassBuilder PB(TMPointer, PTOptions, std::nullopt, &PIC,
                 Config.VFS ? Config.VFS : vfs::getRealFileSystem());

  StandardInstrumentations SI(M.getContext(), Config.DebugPassManager,
                              Config.VerifyEach);
  SI.registerCallbacks(PIC, &MAM);

  TargetLibraryInfoImpl TLII(M.getTargetTriple(), TM.Options.VecLib);
  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });
  MAM.registerPass([&] { return MachineModuleAnalysis(MMI); });
  MAM.registerPass([&] {
    const llvm::TargetOptions &Options = TM.Options;
    return RuntimeLibraryAnalysis(Options.MCOptions.ABIName, Options.VecLib);
  });

  if (Config.SummaryIndex)
    MAM.registerPass([&] {
      return ImmutableModuleSummaryIndexAnalysis(Config.SummaryIndex);
    });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);

  Error BuildPipelineError =
      TM.buildCodeGenPipeline(MPM, MAM, OS, DwoOS ? &DwoOS->os() : nullptr,
                              CGFT, Opt, MMI.getContext(), &PIC);
  if (BuildPipelineError)
    return BuildPipelineError;

  MPM.run(M, MAM);
  return Error::success();
}

Error llvm::runCodeGenPipeline(TargetMachine &TM, Module &M,
                               raw_pwrite_stream &OS,
                               std::unique_ptr<ToolOutputFile> &DwoOS,
                               CodeGenFileType CGFT,
                               const CodeGenPipelineConfig &Config) {
  if (ForceNewPM == cl::boolOrDefault::BOU_TRUE ||
      (TM.shouldDefaultToNewPM() && ForceNewPM != cl::boolOrDefault::BOU_FALSE))
    return runCodeGenPipelineNewPM(TM, M, OS, DwoOS, CGFT, Config);

  return runCodeGenPipelineLegacy(TM, M, OS, DwoOS, CGFT, Config);
}
