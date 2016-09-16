//===-LTOBackend.cpp - LLVM Link Time Optimizer Backend -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/LTO/legacy/UpdateCompilerUsed.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"

using namespace llvm;
using namespace lto;

Error Config::addSaveTemps(std::string OutputFileName,
                           bool UseInputModulePath) {
  ShouldDiscardValueNames = false;

  std::error_code EC;
  ResolutionFile = llvm::make_unique<raw_fd_ostream>(
      OutputFileName + "resolution.txt", EC, sys::fs::OpenFlags::F_Text);
  if (EC)
    return errorCodeToError(EC);

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
        PathPrefix = OutputFileName + utostr(Task);
      } else
        PathPrefix = M.getModuleIdentifier();
      std::string Path = PathPrefix + "." + PathSuffix + ".bc";
      std::error_code EC;
      raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::F_None);
      if (EC) {
        // Because -save-temps is a debugging feature, we report the error
        // directly and exit.
        llvm::errs() << "failed to open " << Path << ": " << EC.message()
                     << '\n';
        exit(1);
      }
      WriteBitcodeToFile(&M, OS, /*ShouldPreserveUseListOrder=*/false);
      return true;
    };
  };

  setHook("0.preopt", PreOptModuleHook);
  setHook("1.promote", PostPromoteModuleHook);
  setHook("2.internalize", PostInternalizeModuleHook);
  setHook("3.import", PostImportModuleHook);
  setHook("4.opt", PostOptModuleHook);
  setHook("5.precodegen", PreCodeGenModuleHook);

  CombinedIndexHook = [=](const ModuleSummaryIndex &Index) {
    std::string Path = OutputFileName + "index.bc";
    std::error_code EC;
    raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::F_None);
    if (EC) {
      // Because -save-temps is a debugging feature, we report the error
      // directly and exit.
      llvm::errs() << "failed to open " << Path << ": " << EC.message() << '\n';
      exit(1);
    }
    WriteIndexToFile(Index, OS);
    return true;
  };

  return Error();
}

namespace {

std::unique_ptr<TargetMachine>
createTargetMachine(Config &Conf, StringRef TheTriple,
                    const Target *TheTarget) {
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple(TheTriple));
  for (const std::string &A : Conf.MAttrs)
    Features.AddFeature(A);

  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      TheTriple, Conf.CPU, Features.getString(), Conf.Options, Conf.RelocModel,
      Conf.CodeModel, Conf.CGOptLevel));
}

static void runNewPMCustomPasses(Module &Mod, TargetMachine *TM,
                                 std::string PipelineDesc,
                                 bool DisableVerify) {
  PassBuilder PB(TM);
  AAManager AA;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return std::move(AA); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;

  // Always verify the input.
  MPM.addPass(VerifierPass());

  // Now, add all the passes we've been requested to.
  if (!PB.parsePassPipeline(MPM, PipelineDesc))
    report_fatal_error("unable to parse pass pipeline description: " +
                       PipelineDesc);

  if (!DisableVerify)
    MPM.addPass(VerifierPass());
  MPM.run(Mod, MAM);
}

static void runOldPMPasses(Config &Conf, Module &Mod, TargetMachine *TM,
                           bool IsThinLto) {
  legacy::PassManager passes;
  passes.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM->getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  // Unconditionally verify input since it is not verified before this
  // point and has unknown origin.
  PMB.VerifyInput = true;
  PMB.VerifyOutput = !Conf.DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = Conf.OptLevel;
  if (IsThinLto)
    PMB.populateThinLTOPassManager(passes);
  else
    PMB.populateLTOPassManager(passes);
  passes.run(Mod);
}

bool opt(Config &Conf, TargetMachine *TM, unsigned Task, Module &Mod,
         bool IsThinLto) {
  Mod.setDataLayout(TM->createDataLayout());
  if (Conf.OptPipeline.empty())
    runOldPMPasses(Conf, Mod, TM, IsThinLto);
  else
    runNewPMCustomPasses(Mod, TM, Conf.OptPipeline, Conf.DisableVerify);
  return !Conf.PostOptModuleHook || Conf.PostOptModuleHook(Task, Mod);
}

/// Monolithic LTO does not support caching (yet), this is a convenient wrapper
/// around AddOutput to workaround this.
static AddOutputFn getUncachedOutputWrapper(AddOutputFn &AddOutput,
                                            unsigned Task) {
  return [Task, &AddOutput](unsigned TaskId) {
    auto Output = AddOutput(Task);
    if (Output->isCachingEnabled() && Output->tryLoadFromCache(""))
      report_fatal_error("Cache hit without a valid key?");
    assert(Task == TaskId && "Unexpexted TaskId mismatch");
    return Output;
  };
}

void codegen(Config &Conf, TargetMachine *TM, AddOutputFn AddOutput,
             unsigned Task, Module &Mod) {
  if (Conf.PreCodeGenModuleHook && !Conf.PreCodeGenModuleHook(Task, Mod))
    return;

  auto Output = AddOutput(Task);
  std::unique_ptr<raw_pwrite_stream> OS = Output->getStream();
  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS,
                              TargetMachine::CGFT_ObjectFile))
    report_fatal_error("Failed to setup codegen");
  CodeGenPasses.run(Mod);
}

void splitCodeGen(Config &C, TargetMachine *TM, AddOutputFn AddOutput,
                  unsigned ParallelCodeGenParallelismLevel,
                  std::unique_ptr<Module> Mod) {
  ThreadPool CodegenThreadPool(ParallelCodeGenParallelismLevel);
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  SplitModule(
      std::move(Mod), ParallelCodeGenParallelismLevel,
      [&](std::unique_ptr<Module> MPart) {
        // We want to clone the module in a new context to multi-thread the
        // codegen. We do it by serializing partition modules to bitcode
        // (while still on the main thread, in order to avoid data races) and
        // spinning up new threads which deserialize the partitions into
        // separate contexts.
        // FIXME: Provide a more direct way to do this in LLVM.
        SmallString<0> BC;
        raw_svector_ostream BCOS(BC);
        WriteBitcodeToFile(MPart.get(), BCOS);

        // Enqueue the task
        CodegenThreadPool.async(
            [&](const SmallString<0> &BC, unsigned ThreadId) {
              LTOLLVMContext Ctx(C);
              ErrorOr<std::unique_ptr<Module>> MOrErr = parseBitcodeFile(
                  MemoryBufferRef(StringRef(BC.data(), BC.size()), "ld-temp.o"),
                  Ctx);
              if (!MOrErr)
                report_fatal_error("Failed to read bitcode");
              std::unique_ptr<Module> MPartInCtx = std::move(MOrErr.get());

              std::unique_ptr<TargetMachine> TM =
                  createTargetMachine(C, MPartInCtx->getTargetTriple(), T);

              codegen(C, TM.get(),
                      getUncachedOutputWrapper(AddOutput, ThreadId), ThreadId,
                      *MPartInCtx);
            },
            // Pass BC using std::move to ensure that it get moved rather than
            // copied into the thread's context.
            std::move(BC), ThreadCount++);
      },
      false);
}

Expected<const Target *> initAndLookupTarget(Config &C, Module &Mod) {
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

}

static void handleAsmUndefinedRefs(Module &Mod, TargetMachine &TM) {
  // Collect the list of undefined symbols used in asm and update
  // llvm.compiler.used to prevent optimization to drop these from the output.
  StringSet<> AsmUndefinedRefs;
  object::IRObjectFile::CollectAsmUndefinedRefs(
      Triple(Mod.getTargetTriple()), Mod.getModuleInlineAsm(),
      [&AsmUndefinedRefs](StringRef Name, object::BasicSymbolRef::Flags Flags) {
        if (Flags & object::BasicSymbolRef::SF_Undefined)
          AsmUndefinedRefs.insert(Name);
      });
  updateCompilerUsed(Mod, TM, AsmUndefinedRefs);
}

Error lto::backend(Config &C, AddOutputFn AddOutput,
                   unsigned ParallelCodeGenParallelismLevel,
                   std::unique_ptr<Module> Mod) {
  Expected<const Target *> TOrErr = initAndLookupTarget(C, *Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM =
      createTargetMachine(C, Mod->getTargetTriple(), *TOrErr);

  handleAsmUndefinedRefs(*Mod, *TM);

  if (!C.CodeGenOnly)
    if (!opt(C, TM.get(), 0, *Mod, /*IsThinLto=*/false))
      return Error();

  if (ParallelCodeGenParallelismLevel == 1) {
    codegen(C, TM.get(), getUncachedOutputWrapper(AddOutput, 0), 0, *Mod);
  } else {
    splitCodeGen(C, TM.get(), AddOutput, ParallelCodeGenParallelismLevel,
                 std::move(Mod));
  }
  return Error();
}

Error lto::thinBackend(Config &Conf, unsigned Task, AddOutputFn AddOutput,
                       Module &Mod, ModuleSummaryIndex &CombinedIndex,
                       const FunctionImporter::ImportMapTy &ImportList,
                       const GVSummaryMapTy &DefinedGlobals,
                       MapVector<StringRef, MemoryBufferRef> &ModuleMap) {
  Expected<const Target *> TOrErr = initAndLookupTarget(Conf, Mod);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM =
      createTargetMachine(Conf, Mod.getTargetTriple(), *TOrErr);

  handleAsmUndefinedRefs(Mod, *TM);

  if (Conf.CodeGenOnly) {
    codegen(Conf, TM.get(), AddOutput, Task, Mod);
    return Error();
  }

  if (Conf.PreOptModuleHook && !Conf.PreOptModuleHook(Task, Mod))
    return Error();

  renameModuleForThinLTO(Mod, CombinedIndex);

  thinLTOResolveWeakForLinkerModule(Mod, DefinedGlobals);

  if (Conf.PostPromoteModuleHook && !Conf.PostPromoteModuleHook(Task, Mod))
    return Error();

  if (!DefinedGlobals.empty())
    thinLTOInternalizeModule(Mod, DefinedGlobals);

  if (Conf.PostInternalizeModuleHook &&
      !Conf.PostInternalizeModuleHook(Task, Mod))
    return Error();

  auto ModuleLoader = [&](StringRef Identifier) {
    assert(Mod.getContext().isODRUniquingDebugTypes() &&
           "ODR Type uniquing should be enabled on the context");
    return std::move(getLazyBitcodeModule(MemoryBuffer::getMemBuffer(
                                              ModuleMap[Identifier], false),
                                          Mod.getContext(),
                                          /*ShouldLazyLoadMetadata=*/true)
                         .get());
  };

  FunctionImporter Importer(CombinedIndex, ModuleLoader);
  Importer.importFunctions(Mod, ImportList);

  if (Conf.PostImportModuleHook && !Conf.PostImportModuleHook(Task, Mod))
    return Error();

  if (!opt(Conf, TM.get(), Task, Mod, /*IsThinLto=*/true))
    return Error();

  codegen(Conf, TM.get(), AddOutput, Task, Mod);
  return Error();
}
