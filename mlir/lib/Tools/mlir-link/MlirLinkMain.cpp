//===- MlirLinkMain.cpp - MLIR Link main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-link/MlirLinkMain.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Linker/Linker.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace llvm;

/// This class is intended to manage the handling of command line options for
/// creating a linker config. This is a singleton.
struct LinkerCLOptions : public LinkerConfig {
  /// Returns the command line option category for the linker options
  static cl::OptionCategory &getCategory() {
    static cl::OptionCategory linkerCategory("MLIR Linker Options");
    return linkerCategory;
  }

  /// External storage for input and output file command-line options.
  std::vector<std::string> inputFiles;
  std::string outputFile = "-";

  /// Creates and initializes a LinkerConfig from command line options.
  /// These options are static but use ExternalStorage to initialize the
  /// members of the LinkerConfig class.
  LinkerCLOptions() {
    // Allow operation with no registered dialects.
    // This option is for convenience during testing only and discouraged in
    // general.
    static cl::opt<bool, true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        cl::desc("Allow operation with no registered dialects"),
        cl::location(allowUnregisteredDialectsFlag), cl::init(false),
        cl::cat(getCategory()));

    static cl::opt<bool, true> internalizeLinkedSymbols(
        "internalize", cl::desc("Internalize linked symbols"),
        cl::location(internalizeLinkedSymbolsFlag), cl::init(false),
        cl::cat(getCategory()));

    static cl::opt<bool, true> linkOnlyNeeded(
        "only-needed", cl::desc("Link only needed symbols"),
        cl::location(linkOnlyNeededFlag), cl::init(false),
        cl::cat(getCategory()));

    static cl::list<std::string> clInputFiles(
        cl::Positional, cl::OneOrMore, cl::desc("<input MLIR files>"),
        cl::cat(getCategory()), cl::callback([this](const std::string &val) {
          inputFiles.push_back(val);
        }));

    static cl::opt<std::string> clOutputFile(
        "o", cl::desc("Output filename"), cl::init("-"), cl::cat(getCategory()),
        cl::callback([this](const std::string &val) { outputFile = val; }));
  }

  static void setupAndParse(int argc, char **argv) {
    // Parse command line
    cl::HideUnrelatedOptions({&getCategory(), &getColorCategory()});
    cl::ParseCommandLineOptions(argc, argv, "mlir-link");
  }
};

ManagedStatic<LinkerCLOptions> clOptionsConfig;
LinkerConfig createLinkerConfigFromCLOptions() { return *clOptionsConfig; }

/// Handles the processing of input files for the linker.
/// This class encapsulates all the file handling logic for linker.
class FileProcessor {
public:
  explicit FileProcessor(Linker &linker) : linker(linker) {}

  using FileConfig = Linker::LinkFileConfig;

  /// Process and link multiple input files
  LogicalResult linkFiles(const std::vector<std::string> &fileNames,
                          raw_ostream &os) {
    unsigned flags = linker.getFlags();
    FileConfig config = linker.firstLinkFileConfig(flags);

    for (const auto &fileName : fileNames) {
      std::string errorMessage;
      auto input = openInputFile(fileName, &errorMessage);

      if (!input) {
        return emitFileError(fileName, errorMessage);
      }

      // Process file with potentially multiple modules
      if (failed(processInputFile(std::move(input), config, os)))
        return emitFileError(fileName, "Failed to process input file");

      // Update config for subsequent files
      config = linker.linkFileConfig(flags);
    }

    return success();
  }

private:
  /// Process a single input file, potentially containing multiple modules
  LogicalResult processInputFile(std::unique_ptr<MemoryBuffer> buffer,
                                 FileConfig config, raw_ostream &os) {
    auto sourceMgr = std::make_shared<SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(buffer), SMLoc());

    auto ctx = linker.getContext();
    bool wasThreadingEnabled = ctx->isMultithreadingEnabled();
    ctx->disableMultithreading();

    // Parse the source file
    OwningOpRef<Operation *> op =
        parseSourceFileForTool(sourceMgr, ctx, true /*insertImplicitModule*/);

    ctx->enableMultithreading(wasThreadingEnabled);

    if (!op)
      return emitError("Failed to parse source file");

    // TBD: symbol promotion

    // TBD: internalization

    // Link the parsed module
    return linker.linkInModule(std::move(op), config.flags);
  }

  LogicalResult emitFileError(const Twine &fileName, const Twine &message) {
    return emitError("Error processing file '" + fileName + "': " + message);
  }

  LogicalResult emitError(const Twine &message) {
    return mlir::emitError(mlir::UnknownLoc::get(linker.getContext()), message);
  }

  Linker &linker;
};

LogicalResult mlir::MlirLinkMain(int argc, char **argv,
                                 DialectRegistry &registry) {
  // Initialize LLVM infrastructure
  InitLLVM initLLVM(argc, argv);

  auto config = createLinkerConfigFromCLOptions();

  LinkerCLOptions::setupAndParse(argc, argv);

  MLIRContext context(registry);
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());

  // Create composite module
  auto composite = [&context]() {
    OpBuilder builder(&context);
    return OwningOpRef<ModuleOp>(builder.create<ModuleOp>(
        FileLineColLoc::get(&context, "mlir-link", 0, 0)));
  }();

  auto dst = dyn_cast<LinkableModuleOpInterface>(composite->getOperation());
  if (!dst)
    return composite->emitError(
        "Target module does not support linking. Expected a "
        "LinkableModuleOpInterface.");

  Linker linker(dst, config);
  FileProcessor processor(linker);

  // Prepare output file
  std::string errorMessage;
  auto output = openOutputFile(clOptionsConfig->outputFile, &errorMessage);

  if (!output) {
    errs() << errorMessage;
    return failure();
  }

  // First add all the regular input files
  if (failed(processor.linkFiles(clOptionsConfig->inputFiles, output->os())))
    return failure();

  composite.get()->print(output->os());
  output->keep();

  return success();
}
