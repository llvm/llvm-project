#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> Wave("wave-goodbye", cl::init(false),
                          cl::desc("wave good bye"));

static cl::opt<bool> LastWords("last-words", cl::init(false),
                               cl::desc("say last words (suppress codegen)"));

static cl::opt<bool>
    ByeWarn("bye-warn", cl::init(false),
            cl::desc("emit a warning per function through the frontend, in the "
                     "backend plugin's own warning group"));

namespace {

// A backend (IR-layer) diagnostic that names its own warning group. Overriding
// getWarningGroup() lets the frontend control it with -W<group> just like a
// frontend plugin's diagnostic, instead of the coarse -Wbackend-plugin
// umbrella. Using a plugin diagnostic kind routes it through the frontend's
// generic backend-diagnostic path. By convention the group is "<plugin>-plugin".
class DiagnosticInfoBye : public DiagnosticInfo {
  const Twine &Msg;

public:
  DiagnosticInfoBye(const Twine &Msg LLVM_LIFETIME_BOUND)
      : DiagnosticInfo(getNextAvailablePluginDiagnosticKind(), DS_Warning),
        Msg(Msg) {}
  void print(DiagnosticPrinter &DP) const override { DP << Msg; }
  StringRef getWarningGroup() const override { return "bye-plugin"; }
};

bool runBye(Function &F) {
  if (Wave) {
    errs() << "Bye: ";
    errs().write_escaped(F.getName()) << '\n';
  }
  if (ByeWarn)
    F.getContext().diagnose(
        DiagnosticInfoBye("Bye saw function '" + F.getName() + "'"));
  return false;
}

struct LegacyBye : public FunctionPass {
  static char ID;
  LegacyBye() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override { return runBye(F); }
};

struct Bye : OptionalPassInfoMixin<Bye> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (!runBye(F))
      return PreservedAnalyses::all();
    return PreservedAnalyses::none();
  }
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
}

bool preCodeGenCallback(Module &M, TargetMachine &, CodeGenFileType CGFT,
                        raw_pwrite_stream &OS) {
  if (LastWords) {
    if (CGFT != CodeGenFileType::AssemblyFile) {
      // Test error emission.
      M.getContext().emitError("last words unsupported for binary output");
      return false;
    }
    OS << "CodeGen Bye\n";
    return true; // Suppress remaining compilation pipeline.
  }
  // Do nothing.
  return false;
}

} // namespace

char LegacyBye::ID = 0;

static RegisterPass<LegacyBye> X("goodbye", "Good Bye World Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

/* New PM Registration */
llvm::PassPluginLibraryInfo getByePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Bye", LLVM_VERSION_STRING,
          registerPassBuilderCallbacks, preCodeGenCallback};
}

#ifndef LLVM_BYE_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getByePluginInfo();
}
#endif
