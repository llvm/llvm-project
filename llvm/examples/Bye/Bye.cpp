#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> Wave("wave-goodbye", cl::init(false),
                          cl::desc("wave good bye"));

static cl::opt<bool> PrintEntryPointCallbacks(
    "print-ep-callbacks", cl::init(false),
    cl::desc("Print names of all entry-points upon callback"));

namespace {

bool runBye(Function &F) {
  if (Wave) {
    errs() << "Bye: ";
    errs().write_escaped(F.getName()) << '\n';
  }
  return false;
}

struct LegacyBye : public FunctionPass {
  static char ID;
  LegacyBye() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override { return runBye(F); }
};

struct Bye : PassInfoMixin<Bye> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (!runBye(F))
      return PreservedAnalyses::all();
    return PreservedAnalyses::none();
  }
};

struct PrintStage : PassInfoMixin<Bye> {
  PrintStage(std::string Name) : Name(std::move(Name)) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    outs() << Name << "\n";
    return PreservedAnalyses::none();
  }
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    outs() << Name << "\n";
    return PreservedAnalyses::none();
  }

private:
  std::string Name;
};

void registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerVectorizerStartEPCallback(
      [](llvm::FunctionPassManager &PM, OptimizationLevel Level) {
        PM.addPass(Bye());
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, llvm::FunctionPassManager &PM,
         ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "goodbye") {
          PM.addPass(Bye());
          return true;
        }
        return false;
      });

  if (PrintEntryPointCallbacks) {
    PB.registerPipelineStartEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(PrintStage("PipelineStart"));
          return true;
        });

    PB.registerPipelineEarlySimplificationEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt,
           ThinOrFullLTOPhase Phase) {
          MPM.addPass(PrintStage("PipelineEarlySimplification"));
          return true;
        });

    PB.registerOptimizerEarlyEPCallback([](ModulePassManager &MPM,
                                           OptimizationLevel Opt,
                                           ThinOrFullLTOPhase Phase) {
      MPM.addPass(PrintStage("OptimizerEarly"));
      return true;
    });

    PB.registerOptimizerLastEPCallback([](ModulePassManager &MPM,
                                          OptimizationLevel Opt,
                                          ThinOrFullLTOPhase Phase) {
      MPM.addPass(PrintStage("OptimizerLast"));
      return true;
    });

    PB.registerPeepholeEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(PrintStage("Peephole"));
          return true;
        });

    PB.registerScalarOptimizerLateEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(PrintStage("ScalarOptimizerLate"));
          return true;
        });

    PB.registerVectorizerStartEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(PrintStage("VectorizerStart"));
          return true;
        });

    PB.registerVectorizerEndEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(PrintStage("VectorizerEnd"));
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(PrintStage("FullLinkTimeOptimizationEarly"));
          return true;
        });

    PB.registerFullLinkTimeOptimizationLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(PrintStage("FullLinkTimeOptimizationLast"));
          return true;
        });
  }
}

} // namespace

char LegacyBye::ID = 0;

static RegisterPass<LegacyBye> X("goodbye", "Good Bye World Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

/* New PM Registration */
llvm::PassPluginLibraryInfo getByePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Bye", LLVM_VERSION_STRING,
          registerPassBuilderCallbacks};
}

#ifndef LLVM_BYE_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getByePluginInfo();
}
#endif
