#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/raw_ostream.h"

#define OPTIONS_STRUCT_DECL
#include "ByeOptions.inc"
#define OPTIONS_STRUCT_DEFS
#include "ByeOptions.inc"

using namespace llvm;

// Hosts that load the plugin before they parse their command line accept the
// options directly, e.g. clang with -mllvm.
LLVM_REGISTER_LIBRARY_OPTIONS(llvm::ByeOptions)

namespace {

// The host copies the options into the context of each compilation
const ByeOptions &getOptions(const LLVMContext &Ctx) {
  return Ctx.getLibraryOptions().get<ByeOptions>();
}

bool runBye(Function &F) {
  const ByeOptions &Opts = getOptions(F.getContext());
  if (Opts.wave_goodbye) {
    errs() << "Bye: ";
    errs().write_escaped(F.getName()) << '\n';
  }
  if (!Opts.bye_greeting.empty()) {
    errs() << Opts.bye_greeting << ": ";
    errs().write_escaped(F.getName()) << '\n';
  }
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

bool preCodeGenCallback(TargetMachine &, Module &M, raw_pwrite_stream &OS,
                        raw_pwrite_stream *DwoOS, CodeGenFileType CGFT, bool,
                        bool, const ModuleSummaryIndex *,
                        IntrusiveRefCntPtr<vfs::FileSystem>) {
  if (getOptions(M.getContext()).last_words) {
    if (CGFT != CodeGenFileType::AssemblyFile) {
      // Test error emission.
      M.getContext().emitError("last words unsupported for binary output");
      return false;
    }
    OS << "CodeGen Bye\n";
    if (DwoOS)
      *DwoOS << "DwoGen Bye\n";
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

static Error parseArguments(cl::LibraryOptionsContext &LibraryOptions,
                            ArrayRef<const char *> Args) {
  // Parse into the host's options, where -mllvm options end up as well.
  // Unlike cl::ParseCommandLineOptions, this doesn't involve the option
  // registry of the host, so the host's own options and positional arguments
  // can't interfere.
  static constexpr opt::LibraryOptionsParser Parser{
      std::in_place_type<ByeOptions>};
  for (unsigned I = 0, Consumed = 0; I < Args.size(); I += Consumed)
    if (Error E = LibraryOptions.parse(Parser, Args.drop_front(I), Consumed))
      return E;
  return Error::success();
}

/* New PM Registration */
llvm::PassPluginLibraryInfo getByePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Bye",
          LLVM_VERSION_STRING,     registerPassBuilderCallbacks,
          preCodeGenCallback,      parseArguments};
}

#ifndef LLVM_BYE_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getByePluginInfo();
}
#endif
