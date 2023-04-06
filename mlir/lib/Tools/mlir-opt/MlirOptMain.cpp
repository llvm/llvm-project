//===- MlirOptMain.cpp - MLIR Optimizer Driver ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that runs an optimization pass and prints the result back
// out. It is designed to support unit testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectPlugin.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassPlugin.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

namespace {
/// This class is intended to manage the handling of command line options for
/// creating a *-opt config. This is a singleton.
struct MlirOptMainConfigCLOptions : public MlirOptMainConfig {
  MlirOptMainConfigCLOptions() {
    // These options are static but all uses ExternalStorage to initialize the
    // members of the parent class. This is unusual but since this class is a
    // singleton it basically attaches command line option to the singleton
    // members.

    static cl::opt<bool, /*ExternalStorage=*/true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        cl::desc("Allow operation with no registered dialects"),
        cl::location(allowUnregisteredDialectsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> dumpPassPipeline(
        "dump-pass-pipeline", cl::desc("Print the pipeline that will be run"),
        cl::location(dumpPassPipelineFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> emitBytecode(
        "emit-bytecode", cl::desc("Emit bytecode when generating output"),
        cl::location(emitBytecodeFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> explicitModule(
        "no-implicit-module",
        cl::desc("Disable implicit addition of a top-level module op during "
                 "parsing"),
        cl::location(useExplicitModuleFlag), cl::init(false));

    static cl::opt<std::string, /*ExternalStorage=*/true> logActionsTo{
        "log-actions-to",
        cl::desc("Log action execution to a file, or stderr if "
                 " '-' is passed"),
        cl::location(logActionsToFlag)};

    static cl::opt<bool, /*ExternalStorage=*/true> showDialects(
        "show-dialects",
        cl::desc("Print the list of registered dialects and exit"),
        cl::location(showDialectsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> splitInputFile(
        "split-input-file",
        cl::desc("Split the input file into pieces and process each "
                 "chunk independently"),
        cl::location(splitInputFileFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyDiagnostics(
        "verify-diagnostics",
        cl::desc("Check that emitted diagnostics match "
                 "expected-* lines on the corresponding line"),
        cl::location(verifyDiagnosticsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyPasses(
        "verify-each",
        cl::desc("Run the verifier after each transformation pass"),
        cl::location(verifyPassesFlag), cl::init(true));

    static cl::list<std::string> passPlugins(
        "load-pass-plugin", cl::desc("Load passes from plugin library"));
    /// Set the callback to load a pass plugin.
    passPlugins.setCallback([&](const std::string &pluginPath) {
      auto plugin = PassPlugin::load(pluginPath);
      if (!plugin) {
        errs() << "Failed to load passes from '" << pluginPath
               << "'. Request ignored.\n";
        return;
      }
      plugin.get().registerPassRegistryCallbacks();
    });

    static cl::list<std::string> dialectPlugins(
        "load-dialect-plugin", cl::desc("Load dialects from plugin library"));
    this->dialectPlugins = std::addressof(dialectPlugins);

    static PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);
  }

  /// Set the callback to load a dialect plugin.
  void setDialectPluginsCallback(DialectRegistry &registry);

  /// Pointer to static dialectPlugins variable in constructor, needed by
  /// setDialectPluginsCallback(DialectRegistry&).
  cl::list<std::string> *dialectPlugins = nullptr;
};
} // namespace

ManagedStatic<MlirOptMainConfigCLOptions> clOptionsConfig;

void MlirOptMainConfig::registerCLOptions(DialectRegistry &registry) {
  clOptionsConfig->setDialectPluginsCallback(registry);
}

MlirOptMainConfig MlirOptMainConfig::createFromCLOptions() {
  return *clOptionsConfig;
}

MlirOptMainConfig &MlirOptMainConfig::setPassPipelineParser(
    const PassPipelineCLParser &passPipeline) {
  passPipelineCallback = [&](PassManager &pm) {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(pm.getContext())) << msg;
      return failure();
    };
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
    if (this->shouldDumpPassPipeline()) {

      pm.dump();
      llvm::errs() << "\n";
    }
    return success();
  };
  return *this;
}

void MlirOptMainConfigCLOptions::setDialectPluginsCallback(
    DialectRegistry &registry) {
  dialectPlugins->setCallback([&](const std::string &pluginPath) {
    auto plugin = DialectPlugin::load(pluginPath);
    if (!plugin) {
      errs() << "Failed to load dialect plugin from '" << pluginPath
             << "'. Request ignored.\n";
      return;
    };
    plugin.get().registerDialectRegistryCallbacks(registry);
  });
}

/// Set the ExecutionContext on the context and handle the observers.
class InstallDebugHandler {
public:
  InstallDebugHandler(MLIRContext &context, const MlirOptMainConfig &config) {
    if (config.getLogActionsTo().empty()) {
      if (tracing::DebugCounter::isActivated())
        context.registerActionHandler(tracing::DebugCounter());
      return;
    }
    if (tracing::DebugCounter::isActivated())
      emitError(UnknownLoc::get(&context),
                "Debug counters are incompatible with --log-actions-to option "
                "and are disabled");
    std::string errorMessage;
    logActionsFile = openOutputFile(config.getLogActionsTo(), &errorMessage);
    if (!logActionsFile) {
      emitError(UnknownLoc::get(&context),
                "Opening file for --log-actions-to failed: ")
          << errorMessage << "\n";
      return;
    }
    logActionsFile->keep();
    raw_fd_ostream &logActionsStream = logActionsFile->os();
    actionLogger = std::make_unique<tracing::ActionLogger>(logActionsStream);

    executionContext.registerObserver(actionLogger.get());
    context.registerActionHandler(executionContext);
  }

private:
  std::unique_ptr<llvm::ToolOutputFile> logActionsFile;
  std::unique_ptr<tracing::ActionLogger> actionLogger;
  tracing::ExecutionContext executionContext;
};

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult
performActions(raw_ostream &os,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
               MLIRContext *context, const MlirOptMainConfig &config) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Prepare the parser config, and attach any useful/necessary resource
  // handlers. Unhandled external resources are treated as passthrough, i.e.
  // they are not processed and will be emitted directly to the output
  // untouched.
  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(context, /*verifyAfterParse=*/true,
                           &fallbackResourceMap);
  reproOptions.attachResourceParser(parseConfig);

  // Parse the input file and reset the context threading state.
  TimingScope parserTiming = timing.nest("Parser");
  OwningOpRef<Operation *> op = parseSourceFileForTool(
      sourceMgr, parseConfig, !config.shouldUseExplicitModule());
  context->enableMultithreading(wasThreadingEnabled);
  if (!op)
    return failure();
  parserTiming.stop();

  // Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (failed(reproOptions.apply(pm)) || failed(config.setupPassPipeline(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();

  // Print the output.
  TimingScope outputTiming = timing.nest("Output");
  if (config.shouldEmitBytecode()) {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);
    writeBytecodeToFile(op.get(), os, writerConfig);
  } else {
    AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                      &fallbackResourceMap);
    op.get()->print(os, asmState);
    os << '\n';
  }
  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult processBuffer(raw_ostream &os,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   const MlirOptMainConfig &config,
                                   DialectRegistry &registry,
                                   llvm::ThreadPool *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool)
    context.setThreadPool(*threadPool);

  // Parse the input file.
  if (config.shouldPreloadDialectsInContext())
    context.loadAllAvailableDialects();
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  if (config.shouldVerifyDiagnostics())
    context.printOpOnDiagnostic(false);

  InstallDebugHandler installDebugHandler(context, config);

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!config.shouldVerifyDiagnostics()) {
    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
    return performActions(os, sourceMgr, &context, config);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(*sourceMgr, &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, sourceMgr, &context, config);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

LogicalResult mlir::MlirOptMain(llvm::raw_ostream &outputStream,
                                std::unique_ptr<llvm::MemoryBuffer> buffer,
                                DialectRegistry &registry,
                                const MlirOptMainConfig &config) {
  if (config.shouldShowDialects()) {
    llvm::outs() << "Available Dialects: ";
    interleave(registry.getDialectNames(), llvm::outs(), ",");
    llvm::outs() << "\n";
  }

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPool *threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    return processBuffer(os, std::move(chunkBuffer), config, registry,
                         threadPool);
  };
  return splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream,
                               config.shouldSplitInputFile(),
                               /*insertMarkerInOutput=*/true);
}

LogicalResult mlir::MlirOptMain(raw_ostream &outputStream,
                                std::unique_ptr<MemoryBuffer> buffer,
                                PassPipelineFn passManagerSetupFn,
                                DialectRegistry &registry, bool splitInputFile,
                                bool verifyDiagnostics, bool verifyPasses,
                                bool allowUnregisteredDialects,
                                bool preloadDialectsInContext,
                                bool emitBytecode, bool explicitModule) {
  return MlirOptMain(outputStream, std::move(buffer), registry,
                     MlirOptMainConfig{}
                         .splitInputFile(splitInputFile)
                         .verifyDiagnostics(verifyDiagnostics)
                         .verifyPasses(verifyPasses)
                         .allowUnregisteredDialects(allowUnregisteredDialects)
                         .preloadDialectsInContext(preloadDialectsInContext)
                         .emitBytecode(emitBytecode)
                         .useExplicitModule(explicitModule)
                         .setPassPipelineSetupFn(passManagerSetupFn));
}

LogicalResult mlir::MlirOptMain(
    raw_ostream &outputStream, std::unique_ptr<MemoryBuffer> buffer,
    const PassPipelineCLParser &passPipeline, DialectRegistry &registry,
    bool splitInputFile, bool verifyDiagnostics, bool verifyPasses,
    bool allowUnregisteredDialects, bool preloadDialectsInContext,
    bool emitBytecode, bool explicitModule, bool dumpPassPipeline) {
  return MlirOptMain(outputStream, std::move(buffer), registry,
                     MlirOptMainConfig{}
                         .splitInputFile(splitInputFile)
                         .verifyDiagnostics(verifyDiagnostics)
                         .verifyPasses(verifyPasses)
                         .allowUnregisteredDialects(allowUnregisteredDialects)
                         .preloadDialectsInContext(preloadDialectsInContext)
                         .emitBytecode(emitBytecode)
                         .useExplicitModule(explicitModule)
                         .dumpPassPipeline(dumpPassPipeline)
                         .setPassPipelineParser(passPipeline));
}

LogicalResult mlir::MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                                DialectRegistry &registry,
                                bool preloadDialectsInContext) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  InitLLVM y(argc, argv);

  // Register any command line options.
  MlirOptMainConfig::registerCLOptions(registry);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();
  config.preloadDialectsInContext(preloadDialectsInContext);

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}
