#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

static cl::opt<bool> Wave("wave-goodbye", cl::init(false),
                          cl::desc("wave good bye"));

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

} // namespace

char LegacyBye::ID = 0;

static RegisterPass<LegacyBye> X("goodbye", "Good Bye World Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

/* Legacy PM Registration */
static llvm::RegisterStandardPasses RegisterBye(
    llvm::PassManagerBuilder::EP_EarlyAsPossible,
    [](const llvm::PassManagerBuilder &Builder,
       llvm::legacy::PassManagerBase &PM) { PM.add(new LegacyBye()); });

/* New PM Registration */
llvm::PassPluginLibraryInfo getByePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Bye", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerVectorizerStartEPCallback(
                [](llvm::FunctionPassManager &PM,
                   llvm::PassBuilder::OptimizationLevel Level) {
                  PM.addPass(Bye());
                });
          }};
}

#ifndef LLVM_BYE_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getByePluginInfo();
}
#endif
