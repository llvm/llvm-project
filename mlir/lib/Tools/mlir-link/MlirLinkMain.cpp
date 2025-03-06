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
#include "mlir/IR/Verifier.h" // TODO: Remove
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
using namespace mlir::link;

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
  using OwningMemoryBuffer = std::unique_ptr<MemoryBuffer>;

  explicit FileProcessor(Linker &linker, raw_ostream &os, StringRef inMarker,
                         StringRef outMarker)
      : linker(linker), os(os), inMarker(inMarker), outMarker(outMarker) {}

  /// Process and link multiple input files
  LogicalResult linkFiles(const std::vector<std::string> fileNames) {
    for (StringRef fileName : fileNames) {
      if (failed(processFile(fileName)))
        return failure();
    }

    return success();
  }

private:
  LogicalResult processFile(StringRef fileName) {
    std::string errorMessage;
    auto input = openInputFile(fileName, &errorMessage);
    if (!input)
      return linker.emitFileError(fileName, errorMessage);

    // Process each file chunk
    if (failed(processFile(std::move(input))))
      return linker.emitFileError(fileName, "Failed to process input file");

    return success();
  }

  /// Process a single input file, potentially containing multiple modules
  LogicalResult processFile(OwningMemoryBuffer file) {
    return splitAndProcessBuffer(
        std::move(file),
        [this](OwningMemoryBuffer chunk, raw_ostream & /* os */) {
          return processFileChunk(std::move(chunk));
        },
        os, inMarker, outMarker);
  }

  LogicalResult processFileChunk(OwningMemoryBuffer buffer) {
    auto sourceMgr = std::make_shared<SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(buffer), SMLoc());

    MLIRContext *ctx = linker.getContext();
    bool wasThreadingEnabled = ctx->isMultithreadingEnabled();
    ctx->disableMultithreading();

    // Parse the source file
    OwningOpRef<Operation *> op =
        parseSourceFileForTool(sourceMgr, ctx, /*insertImplicitModule=*/true);
    ctx->enableMultithreading(wasThreadingEnabled);

    if (!op)
      return linker.emitError("Failed to parse source file");

    if (!isa<ModuleOp>(op.get()))
      return op->emitError("Expected a ModuleOp");

    // TBD: symbol promotion

    // TBD: internalization
    OwningOpRef<ModuleOp> mod = cast<ModuleOp>(op.release());
    // Link the parsed operation
    return linker.linkInModule(std::move(mod));
  }

  Linker &linker;
  raw_ostream &os;
  StringRef inMarker;
  StringRef outMarker;
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

  Linker linker(config, &context);

  // Prepare output file
  std::string errorMessage;
  auto out = openOutputFile(config.outputFile, &errorMessage);

  if (!out) {
    return linker.emitError("Failed to open output file: " + errorMessage);
  }

  StringRef inMarker = config.inputSplitMarker();
  StringRef outMarker = config.outputSplitMarker();

  FileProcessor proc(linker, out->os(), inMarker, outMarker);

  // First add all the regular input files
  if (failed(proc.linkFiles(config.inputFiles)))
    return failure();

  OwningOpRef<ModuleOp> composite = linker.takeModule();
  if (failed(verify(composite.get(), true))) {
    return composite->emitError("verification after linking failed");
  }

  composite->print(out->os());
  out->keep();

  return success();
}
