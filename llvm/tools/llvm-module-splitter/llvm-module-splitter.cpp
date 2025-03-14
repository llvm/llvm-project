
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/ModuleSplitter/ModuleSplitter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;

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

  LLVMModuleAndContext Module;
  return 0;
  //ErrorOrSuccess err = module.create(
  //    [&](LLVMContext &ctx) -> M::ErrorOr<std::unique_ptr<Module>> {
  //      if (std::unique_ptr<Module> module =
  //              readModule(ctx, clOptions.inputFilename))
  //        return module;
  //      return M::Error("could not load LLVM file");
  //    });
  //if (err) {
  //  llvm::errs() << err.getError() << "\n";
  //  return -1;
  //}

  //std::unique_ptr<llvm::ToolOutputFile> output = nullptr;
  //if (clOptions.outputPrefix == "-") {
  //  std::error_code error;
  //  output = std::make_unique<llvm::ToolOutputFile>(
  //      clOptions.outputPrefix, error, llvm::sys::fs::OF_None);
  //  if (error)
  //    exit(clOptions.options.reportError("Cannot open output file: '" +
  //                                       clOptions.outputPrefix +
  //                                       "':" + error.message()));
  //}

  //auto outputLambda =
  //    [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
  //        std::optional<int64_t> idx, unsigned numFunctionsBase) mutable {
  //      LLVMModuleAndContext subModule = produceModule();
  //      if (clOptions.outputPrefix == "-") {
  //        output->os() << "##############################################\n";
  //        if (idx)
  //          output->os() << "# [LLVM Module Split: submodule " << *idx << "]\n";
  //        else
  //          output->os() << "# [LLVM Module Split: main module]\n";
  //        output->os() << "##############################################\n";
  //        output->os() << *subModule;
  //        output->os() << "\n";
  //      } else {
  //        std::string outPath;
  //        if (!idx) {
  //          outPath = clOptions.outputPrefix + ".ll";
  //        } else {
  //          outPath =
  //              (clOptions.outputPrefix + "." + Twine(*idx) + ".ll").str();
  //        }
  //        auto outFile = mlir::openOutputFile(outPath);
  //        if (!outFile) {
  //          exit(clOptions.options.reportError("Cannot open output file: '" +
  //                                             outPath + "."));
  //        }
  //        outFile->os() << *subModule;
  //        outFile->keep();
  //        llvm::outs() << "Write llvm module to " << outPath << "\n";
  //      }
  //    };

  //llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;
  //if (clOptions.perFunctionSplit)
  //  splitPerFunction(std::move(module), outputLambda, symbolLinkageTypes);
  //else
  //  splitPerExported(std::move(module), outputLambda);

  //if (output)
  //  output->keep();
  //return 0;
}
