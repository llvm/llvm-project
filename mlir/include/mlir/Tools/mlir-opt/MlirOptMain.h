//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
#define MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace mlir {
class DialectRegistry;
class PassPipelineCLParser;
class PassManager;

/// Configuration options for the mlir-opt tool.
/// This is intended to help building tools like mlir-opt by collecting the
/// supported options.
/// The API is fluent, and the options are sorted in alphabetical order below.
/// The options can be exposed to the LLVM command line by registering them
/// with `MlirOptMainConfig::registerCLOptions();` and creating a config using
/// `auto config = MlirOptMainConfig::createFromCLOptions();`.
class MlirOptMainConfig {
public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions();

  /// Create a new config with the default set from the CL options.
  static MlirOptMainConfig createFromCLOptions();

  ///
  /// Options.
  ///

  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  MlirOptMainConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  /// Print the pass-pipeline as text before executing.
  MlirOptMainConfig &dumpPassPipeline(bool dump) {
    dumpPassPipelineFlag = dump;
    return *this;
  }
  bool shouldDumpPassPipeline() const { return dumpPassPipelineFlag; }

  /// Set the output format to bytecode instead of textual IR.
  MlirOptMainConfig &emitBytecode(bool emit) {
    emitBytecodeFlag = emit;
    return *this;
  }
  bool shouldEmitBytecode() const { return emitBytecodeFlag; }

  /// Set the filename to use for logging actions, use "-" for stdout.
  MlirOptMainConfig &logActionsTo(StringRef filename) {
    logActionsToFlag = filename;
    return *this;
  }
  /// Get the filename to use for logging actions.
  StringRef getLogActionsTo() const { return logActionsToFlag; }

  /// Set the callback to populate the pass manager.
  MlirOptMainConfig &
  setPassPipelineSetupFn(std::function<LogicalResult(PassManager &)> callback) {
    passPipelineCallback = std::move(callback);
    return *this;
  }

  /// Set the parser to use to populate the pass manager.
  MlirOptMainConfig &setPassPipelineParser(const PassPipelineCLParser &parser);

  /// Populate the passmanager, if any callback was set.
  LogicalResult setupPassPipeline(PassManager &pm) const {
    if (passPipelineCallback)
      return passPipelineCallback(pm);
    return success();
  }

  // Deprecated.
  MlirOptMainConfig &preloadDialectsInContext(bool preload) {
    preloadDialectsInContextFlag = preload;
    return *this;
  }
  bool shouldPreloadDialectsInContext() const {
    return preloadDialectsInContextFlag;
  }

  /// Show the registered dialects before trying to load the input file.
  MlirOptMainConfig &showDialects(bool show) {
    showDialectsFlag = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialectsFlag; }

  /// Set whether to split the input file based on the `// -----` marker into
  /// pieces and process each chunk independently.
  MlirOptMainConfig &splitInputFile(bool split = true) {
    splitInputFileFlag = split;
    return *this;
  }
  bool shouldSplitInputFile() const { return splitInputFileFlag; }

  /// Disable implicit addition of a top-level module op during parsing.
  MlirOptMainConfig &useExplicitModule(bool useExplicitModule) {
    useExplicitModuleFlag = useExplicitModule;
    return *this;
  }
  bool shouldUseExplicitModule() const { return useExplicitModuleFlag; }

  /// Set whether to check that emitted diagnostics match `expected-*` lines on
  /// the corresponding line. This is meant for implementing diagnostic tests.
  MlirOptMainConfig &verifyDiagnostics(bool verify) {
    verifyDiagnosticsFlag = verify;
    return *this;
  }
  bool shouldVerifyDiagnostics() const { return verifyDiagnosticsFlag; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &verifyPasses(bool verify) {
    verifyPassesFlag = verify;
    return *this;
  }
  bool shouldVerifyPasses() const { return verifyPassesFlag; }

protected:
  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  bool allowUnregisteredDialectsFlag = false;

  /// Print the pipeline that will be run.
  bool dumpPassPipelineFlag = false;

  /// Emit bytecode instead of textual assembly when generating output.
  bool emitBytecodeFlag = false;

  /// Log action execution to the given file (or "-" for stdout)
  std::string logActionsToFlag;

  /// The callback to populate the pass manager.
  std::function<LogicalResult(PassManager &)> passPipelineCallback;

  /// Deprecated.
  bool preloadDialectsInContextFlag = false;

  /// Show the registered dialects before trying to load the input file.
  bool showDialectsFlag = false;

  /// Split the input file based on the `// -----` marker into pieces and
  /// process each chunk independently.
  bool splitInputFileFlag = false;

  /// Use an explicit top-level module op during parsing.
  bool useExplicitModuleFlag = false;

  /// Set whether to check that emitted diagnostics match `expected-*` lines on
  /// the corresponding line. This is meant for implementing diagnostic tests.
  bool verifyDiagnosticsFlag = false;

  /// Run the verifier after each transformation pass.
  bool verifyPassesFlag = true;
};

/// This defines the function type used to setup the pass manager. This can be
/// used to pass in a callback to setup a default pass pipeline to be applied on
/// the loaded IR.
using PassPipelineFn = llvm::function_ref<LogicalResult(PassManager &pm)>;

/// Perform the core processing behind `mlir-opt`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          DialectRegistry &registry,
                          const MlirOptMainConfig &config);

/// Perform the core processing behind `mlir-opt`.
/// This API is deprecated, use the MlirOptMainConfig version above instead.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          const PassPipelineCLParser &passPipeline,
                          DialectRegistry &registry, bool splitInputFile,
                          bool verifyDiagnostics, bool verifyPasses,
                          bool allowUnregisteredDialects,
                          bool preloadDialectsInContext = false,
                          bool emitBytecode = false, bool explicitModule = true,
                          bool dumpPassPipeline = false);

/// Perform the core processing behind `mlir-opt`.
/// This API is deprecated, use the MlirOptMainConfig version above instead.
LogicalResult MlirOptMain(
    llvm::raw_ostream &outputStream, std::unique_ptr<llvm::MemoryBuffer> buffer,
    PassPipelineFn passManagerSetupFn, DialectRegistry &registry,
    bool splitInputFile, bool verifyDiagnostics, bool verifyPasses,
    bool allowUnregisteredDialects, bool preloadDialectsInContext = false,
    bool emitBytecode = false, bool explicitModule = true);

/// Implementation for tools like `mlir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - preloadDialectsInContext will trigger the upfront loading of all
///   dialects from the global registry in the MLIRContext. This option is
///   deprecated and will be removed soon.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                          DialectRegistry &registry,
                          bool preloadDialectsInContext = false);

/// Helper wrapper to return the result of MlirOptMain directly from main.
///
/// Example:
///
///     int main(int argc, char **argv) {
///       // ...
///       return mlir::asMainReturnCode(mlir::MlirOptMain(
///           argc, argv, /* ... */);
///     }
///
inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace mlir

#endif // MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
