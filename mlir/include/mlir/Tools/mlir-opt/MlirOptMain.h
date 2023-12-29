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

#include "mlir/Debug/CLOptionsSetup.h"
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
/// with `MlirOptMainConfig::registerCLOptions(DialectRegistry &);` and creating
/// a config using `auto config = MlirOptMainConfig::createFromCLOptions();`.
class MlirOptMainConfig {
public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions(DialectRegistry &dialectRegistry);

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

  /// Set the debug configuration to use.
  MlirOptMainConfig &setDebugConfig(tracing::DebugConfig config) {
    debugConfig = std::move(config);
    return *this;
  }
  tracing::DebugConfig &getDebugConfig() { return debugConfig; }
  const tracing::DebugConfig &getDebugConfig() const { return debugConfig; }

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
  bool shouldElideResourceDataFromBytecode() const {
    return elideResourceDataFromBytecodeFlag;
  }

  /// Set the IRDL file to load before processing the input.
  MlirOptMainConfig &setIrdlFile(StringRef file) {
    irdlFileFlag = file;
    return *this;
  }
  StringRef getIrdlFile() const { return irdlFileFlag; }

  /// Set the bytecode version to emit.
  MlirOptMainConfig &setEmitBytecodeVersion(int64_t version) {
    emitBytecodeVersion = version;
    return *this;
  }
  std::optional<int64_t> bytecodeVersionToEmit() const {
    return emitBytecodeVersion;
  }

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

  /// Enable running the reproducer information stored in resources (if
  /// present).
  MlirOptMainConfig &runReproducer(bool enableReproducer) {
    runReproducerFlag = enableReproducer;
    return *this;
  };

  /// Return true if the reproducer should be run.
  bool shouldRunReproducer() const { return runReproducerFlag; }

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

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &verifyRoundtrip(bool verify) {
    verifyRoundtripFlag = verify;
    return *this;
  }
  bool shouldVerifyRoundtrip() const { return verifyRoundtripFlag; }

protected:
  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  bool allowUnregisteredDialectsFlag = false;

  /// Configuration for the debugging hooks.
  tracing::DebugConfig debugConfig;

  /// Print the pipeline that will be run.
  bool dumpPassPipelineFlag = false;

  /// Emit bytecode instead of textual assembly when generating output.
  bool emitBytecodeFlag = false;

  /// Elide resources when generating bytecode.
  bool elideResourceDataFromBytecodeFlag = false;

  /// Enable the Debugger action hook: Debugger can intercept MLIR Actions.
  bool enableDebuggerActionHookFlag = false;

  /// IRDL file to register before processing the input.
  std::string irdlFileFlag = "";

  /// Location Breakpoints to filter the action logging.
  std::vector<tracing::BreakpointManager *> logActionLocationFilter;

  /// Emit bytecode at given version.
  std::optional<int64_t> emitBytecodeVersion = std::nullopt;

  /// The callback to populate the pass manager.
  std::function<LogicalResult(PassManager &)> passPipelineCallback;

  /// Enable running the reproducer.
  bool runReproducerFlag = false;

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

  /// Verify that the input IR round-trips perfectly.
  bool verifyRoundtripFlag = false;
};

/// This defines the function type used to setup the pass manager. This can be
/// used to pass in a callback to setup a default pass pipeline to be applied on
/// the loaded IR.
using PassPipelineFn = llvm::function_ref<LogicalResult(PassManager &pm)>;

/// Register and parse command line options.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - return std::pair<std::string, std::string> for
///   inputFilename and outputFilename command line option values.
std::pair<std::string, std::string>
registerAndParseCLIOptions(int argc, char **argv, llvm::StringRef toolName,
                           DialectRegistry &registry);

/// Perform the core processing behind `mlir-opt`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          DialectRegistry &registry,
                          const MlirOptMainConfig &config);

/// Implementation for tools like `mlir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                          DialectRegistry &registry);

/// Implementation for tools like `mlir-opt`.
/// This function can be used with registrationAndParseCLIOptions so that
/// CLI options can be accessed before running MlirOptMain.
/// - inputFilename is the name of the input mlir file.
/// - outputFilename is the name of the output file.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef inputFilename,
                          llvm::StringRef outputFilename,
                          DialectRegistry &registry);

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
