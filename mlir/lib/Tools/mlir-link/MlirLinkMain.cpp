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

  /// Show the registered dialects before trying to load the input file.
  LinkerCLOptions &showDialects(bool show) {
    showDialectsFlag = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialectsFlag; }

  /// Set the marker on which to split the input into chunks and process each
  /// chunk independently. Input is not split if empty.
  LinkerCLOptions &
  splitInputFile(std::string splitMarker = kDefaultSplitMarker) {
    splitInputFileFlag = std::move(splitMarker);
    return *this;
  }
  StringRef inputSplitMarker() const { return splitInputFileFlag; }

  /// Set whether to merge the output chunks into one file using the given
  /// marker.
  LinkerCLOptions &
  outputSplitMarker(std::string splitMarker = kDefaultSplitMarker) {
    outputSplitMarkerFlag = std::move(splitMarker);
    return *this;
  }
  StringRef outputSplitMarker() const { return outputSplitMarkerFlag; }

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

    static cl::opt<bool, /*ExternalStorage=*/true> clShowDialects(
        "show-dialects",
        cl::desc("Print the list of registered dialects and exit"),
        cl::location(showDialectsFlag), cl::init(false),
        cl::cat(getCategory()));

    static cl::opt<std::string, /*ExternalStorage=*/true> clSplitInputFile(
        "split-input-file", llvm::cl::ValueOptional,
        cl::desc("Split the input file into chunks using the given or "
                 "default marker and process each chunk independently"),
        cl::callback([&](const std::string &str) {
          // Implicit value: use default marker if flag was used without value.
          if (str.empty())
            clSplitInputFile.setValue(kDefaultSplitMarker);
        }),
        cl::location(splitInputFileFlag), cl::init(""), cl::cat(getCategory()));

    static cl::opt<std::string, /*ExternalStorage=*/true> clOutputSplitMarker(
        "output-split-marker",
        cl::desc("Split marker to use for merging the ouput"),
        cl::location(outputSplitMarkerFlag), cl::init(""),
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

  LinkerCLOptions(const LinkerCLOptions &) = delete;
  LinkerCLOptions(LinkerCLOptions &&) = delete;

  LinkerCLOptions &operator=(const LinkerCLOptions &) = delete;
  LinkerCLOptions &operator=(LinkerCLOptions &&) = delete;

  static void setupAndParse(int argc, char **argv) {
    // Parse command line
    cl::HideUnrelatedOptions({&getCategory(), &getColorCategory()});
    cl::ParseCommandLineOptions(argc, argv, "mlir-link");
  }

  /// Show the registered dialects before trying to load the input file.
  bool showDialectsFlag = false;

  /// Split the input file based on the given marker into chunks and process
  /// each chunk independently. Input is not split if empty.
  std::string splitInputFileFlag = "";

  /// Merge output chunks into one file using the given marker.
  std::string outputSplitMarkerFlag = "";
};

ManagedStatic<LinkerCLOptions> clOptionsConfig;
const LinkerCLOptions &createLinkerConfigFromCLOptions() {
  return *clOptionsConfig;
}

/// Handles the processing of input files for the linker.
/// This class encapsulates all the file handling logic for linker.
class FileProcessor {
public:
  explicit FileProcessor(Linker &linker) : linker(linker) {}

  using FileConfig = Linker::LinkFileConfig;
  using OwningMemoryBuffer = std::unique_ptr<MemoryBuffer>;

  /// Process and link multiple input files
  LogicalResult linkFiles(const std::vector<std::string> inputs,
                          raw_ostream &os, StringRef inMarker,
                          StringRef outMarker) {

    unsigned flags = linker.getFlags();
    FileConfig config = linker.firstLinkFileConfig(flags);

    auto splitAndProcess = [&](OwningMemoryBuffer input) {
      return splitAndProcessBuffer(
          std::move(input),
          [&](OwningMemoryBuffer chunk, raw_ostream &os) {
            return processInputChunk(std::move(chunk), config, os);
          },
          os, inMarker, outMarker);
    };

    for (const auto &file : inputs) {
      std::string errorMessage;
      auto input = openInputFile(file, &errorMessage);

      if (!input)
        return emitFileError(file, errorMessage);

      if (failed(splitAndProcess(std::move(input))))
        return emitFileError(file, "Failed to process input file");

      // Update config for subsequent files
      config = linker.linkFileConfig(flags);
    }

    return success();
  }

private:
  /// Process a single input file, potentially containing multiple modules
  LogicalResult processInputChunk(OwningMemoryBuffer buffer, FileConfig config,
                                  raw_ostream &os) {
    auto sourceMgr = std::make_shared<SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(buffer), SMLoc());

    MLIRContext *ctx = linker.getContext();
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

static LogicalResult printRegisteredDialects(DialectRegistry &registry) {
  llvm::outs() << "Available Dialects: ";
  interleave(registry.getDialectNames(), llvm::outs(), ",");
  llvm::outs() << "\n";
  return success();
}

LogicalResult mlir::MlirLinkMain(int argc, char **argv,
                                 DialectRegistry &registry) {
  // Initialize LLVM infrastructure
  InitLLVM initLLVM(argc, argv);

  const LinkerCLOptions &config = createLinkerConfigFromCLOptions();
  LinkerCLOptions::setupAndParse(argc, argv);

  if (config.shouldShowDialects())
    return printRegisteredDialects(registry);

  MLIRContext context(registry);
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());

  // Create composite module
  OwningOpRef<ModuleOp> composite = [&context]() {
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
  FileProcessor proc(linker);

  // Prepare output file
  std::string errorMessage;
  auto out = openOutputFile(config.outputFile, &errorMessage);

  if (!out) {
    errs() << errorMessage;
    return failure();
  }

  StringRef inMarker = config.inputSplitMarker();
  StringRef outMarker = config.outputSplitMarker();

  // First add all the regular input files
  if (failed(proc.linkFiles(config.inputFiles, out->os(), inMarker, outMarker)))
    return failure();

  composite.get()->print(out->os());
  out->keep();

  return success();
}
