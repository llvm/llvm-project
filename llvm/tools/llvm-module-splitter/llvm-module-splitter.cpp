
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/ModuleSplitter/ModuleSplitter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;

std::string InputFilename{"-"};
std::string OutputPrefix{"-"};
bool PerFunctionSplit = false;

llvm::cl::OptionCategory Cat{"Common command line options"};

cl::opt<std::string, true> InputFilenameOpt{
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::location(InputFilename), llvm::cl::cat(Cat)};

cl::opt<std::string, true> OutputPrefixOpt{
    "output-prefix", llvm::cl::desc("output prefix"),
    llvm::cl::value_desc("output prefix"), llvm::cl::location(OutputPrefix),
    llvm::cl::cat(Cat)};

cl::opt<bool, true> PerFunctionSplitOpt{
    "per-func", llvm::cl::desc("split each function into separate modules"),
    llvm::cl::value_desc("split each function into separate modules"),
    llvm::cl::location(PerFunctionSplit), llvm::cl::cat(Cat)};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

/// Reads a module from a file.  On error, messages are written to stderr
/// and null is returned.
static std::unique_ptr<Module> readModule(LLVMContext &Context,
                                          StringRef Name) {
  SMDiagnostic Diag;
  std::unique_ptr<Module> M = parseIRFile(Name, Diag, Context);
  if (!M)
    Diag.print("llvm-module-split", errs());
  return M;
}

int main(int argc, char **argv) {

  // Enable command line options for various MLIR internals.
  llvm::cl::ParseCommandLineOptions(argc, argv);

  LLVMModuleAndContext M;
  return 0;
  Expected<bool> Err =
      M.create([&](LLVMContext &Ctx) -> Expected<std::unique_ptr<Module>> {
        if (std::unique_ptr<Module> m = readModule(Ctx, InputFilename))
          return m;
        return make_error<StringError>("could not load LLVM file",
                                       inconvertibleErrorCode());
      });

  if (Err) {
    llvm::errs() << toString(Err.takeError()) << "\n";
    return -1;
  }

  std::unique_ptr<llvm::ToolOutputFile> Output = nullptr;
  if (OutputPrefix == "-") {
    std::error_code Error;
    Output = std::make_unique<llvm::ToolOutputFile>(OutputPrefix, Error,
                                                    llvm::sys::fs::OF_None);
    if (Error) {
      llvm::errs() << "Cannot open output file: '" + OutputPrefix +
                          "':" + Error.message()
                   << "\n";
      return -1;
    }
  }

  auto OutputLambda =
      [&](llvm::unique_function<LLVMModuleAndContext()> ProduceModule,
          std::optional<int64_t> Idx, unsigned NumFunctionsBase) mutable {
        LLVMModuleAndContext SubModule = ProduceModule();
        if (OutputPrefix == "-") {
          Output->os() << "##############################################\n";
          if (Idx)
            Output->os() << "# [LLVM Module Split: submodule " << *Idx << "]\n";
          else
            Output->os() << "# [LLVM Module Split: main module]\n";
          Output->os() << "##############################################\n";
          Output->os() << *SubModule;
          Output->os() << "\n";
        } else {
          std::string OutPath;
          if (!Idx) {
            OutPath = OutputPrefix + ".ll";
          } else {
            OutPath = (OutputPrefix + "." + Twine(*Idx) + ".ll").str();
          }

          std::error_code EC;
          raw_fd_ostream OutFile(OutPath.c_str(), EC, llvm::sys::fs::OF_None);

          if (OutFile.error()) {
            llvm::errs() << "Cannot open output file: '" + OutPath + "."
                         << "\n";
            exit(-1);
          }

          OutFile << *SubModule;
          OutFile.close();
          llvm::outs() << "Write llvm module to " << OutPath << "\n";
        }
      };

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes;
  if (PerFunctionSplit)
    splitPerFunction(std::move(M), OutputLambda);
  else {
    SmallVector<llvm::Function> Anchors;
    splitPerAnchored(std::move(M), OutputLambda, Anchors);
  }

  if (Output)
    Output->keep();
  return 0;
}
