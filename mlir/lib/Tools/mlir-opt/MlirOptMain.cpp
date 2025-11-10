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
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Remark/RemarkStreamer.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

namespace {
class BytecodeVersionParser : public cl::parser<std::optional<int64_t>> {
public:
  BytecodeVersionParser(cl::Option &o)
      : cl::parser<std::optional<int64_t>>(o) {}

  bool parse(cl::Option &o, StringRef /*argName*/, StringRef arg,
             std::optional<int64_t> &v) {
    long long w;
    if (getAsSignedInteger(arg, 10, w))
      return o.error("Invalid argument '" + arg +
                     "', only integer is supported.");
    v = w;
    return false;
  }
};

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

    static cl::opt<bool, /*ExternalStorage=*/true> elideResourcesFromBytecode(
        "elide-resource-data-from-bytecode",
        cl::desc("Elide resources when generating bytecode"),
        cl::location(elideResourceDataFromBytecodeFlag), cl::init(false));

    static cl::opt<std::optional<int64_t>, /*ExternalStorage=*/true,
                   BytecodeVersionParser>
        bytecodeVersion(
            "emit-bytecode-version",
            cl::desc("Use specified bytecode when generating output"),
            cl::location(emitBytecodeVersion), cl::init(std::nullopt));

    static cl::opt<std::string, /*ExternalStorage=*/true> irdlFile(
        "irdl-file",
        cl::desc("IRDL file to register before processing the input"),
        cl::location(irdlFileFlag), cl::init(""), cl::value_desc("filename"));

    static cl::opt<VerbosityLevel, /*ExternalStorage=*/true>
        diagnosticVerbosityLevel(
            "mlir-diagnostic-verbosity-level",
            cl::desc("Choose level of diagnostic information"),
            cl::location(diagnosticVerbosityLevelFlag),
            cl::init(VerbosityLevel::ErrorsWarningsAndRemarks),
            cl::values(
                clEnumValN(VerbosityLevel::ErrorsOnly, "errors", "Errors only"),
                clEnumValN(VerbosityLevel::ErrorsAndWarnings, "warnings",
                           "Errors and warnings"),
                clEnumValN(VerbosityLevel::ErrorsWarningsAndRemarks, "remarks",
                           "Errors, warnings and remarks")));

    static cl::opt<bool, /*ExternalStorage=*/true> disableDiagnosticNotes(
        "mlir-disable-diagnostic-notes", cl::desc("Disable diagnostic notes."),
        cl::location(disableDiagnosticNotesFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> explicitModule(
        "no-implicit-module",
        cl::desc("Disable implicit addition of a top-level module op during "
                 "parsing"),
        cl::location(useExplicitModuleFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> listPasses(
        "list-passes", cl::desc("Print the list of registered passes and exit"),
        cl::location(listPassesFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> runReproducer(
        "run-reproducer", cl::desc("Run the pipeline stored in the reproducer"),
        cl::location(runReproducerFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> showDialects(
        "show-dialects",
        cl::desc("Print the list of registered dialects and exit"),
        cl::location(showDialectsFlag), cl::init(false));

    static cl::opt<std::string, /*ExternalStorage=*/true> splitInputFile{
        "split-input-file",
        llvm::cl::ValueOptional,
        cl::callback([&](const std::string &str) {
          // Implicit value: use default marker if flag was used without value.
          if (str.empty())
            splitInputFile.setValue(kDefaultSplitMarker);
        }),
        cl::desc("Split the input file into chunks using the given or "
                 "default marker and process each chunk independently"),
        cl::location(splitInputFileFlag),
        cl::init("")};

    static cl::opt<std::string, /*ExternalStorage=*/true> outputSplitMarker(
        "output-split-marker",
        cl::desc("Split marker to use for merging the ouput"),
        cl::location(outputSplitMarkerFlag), cl::init(kDefaultSplitMarker));

    static cl::opt<SourceMgrDiagnosticVerifierHandler::Level,
                   /*ExternalStorage=*/true>
        verifyDiagnostics{
            "verify-diagnostics", llvm::cl::ValueOptional,
            cl::desc("Check that emitted diagnostics match expected-* lines on "
                     "the corresponding line"),
            cl::location(verifyDiagnosticsFlag),
            cl::values(
                clEnumValN(SourceMgrDiagnosticVerifierHandler::Level::All,
                           "all",
                           "Check all diagnostics (expected, unexpected, "
                           "near-misses)"),
                // Implicit value: when passed with no arguments, e.g.
                // `--verify-diagnostics` or `--verify-diagnostics=`.
                clEnumValN(SourceMgrDiagnosticVerifierHandler::Level::All, "",
                           "Check all diagnostics (expected, unexpected, "
                           "near-misses)"),
                clEnumValN(
                    SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected,
                    "only-expected", "Check only expected diagnostics"))};

    static cl::opt<bool, /*ExternalStorage=*/true> verifyPasses(
        "verify-each",
        cl::desc("Run the verifier after each transformation pass"),
        cl::location(verifyPassesFlag), cl::init(true));

    static cl::opt<bool, /*ExternalStorage=*/true> disableVerifyOnParsing(
        "mlir-very-unsafe-disable-verifier-on-parsing",
        cl::desc("Disable the verifier on parsing (very unsafe)"),
        cl::location(disableVerifierOnParsingFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyRoundtrip(
        "verify-roundtrip",
        cl::desc("Round-trip the IR after parsing and ensure it succeeds"),
        cl::location(verifyRoundtripFlag), cl::init(false));

    static cl::list<std::string> passPlugins(
        "load-pass-plugin", cl::desc("Load passes from plugin library"));

    static cl::opt<std::string, /*ExternalStorage=*/true>
        generateReproducerFile(
            "mlir-generate-reproducer",
            llvm::cl::desc(
                "Generate an mlir reproducer at the provided filename"
                " (no crash required)"),
            cl::location(generateReproducerFileFlag), cl::init(""),
            cl::value_desc("filename"));

    static cl::OptionCategory remarkCategory(
        "Remark Options",
        "Filter remarks by regular expression (llvm::Regex syntax).");

    static llvm::cl::opt<RemarkFormat, /*ExternalStorage=*/true> remarkFormat{
        "remark-format",
        llvm::cl::desc("Specify the format for remark output."),
        cl::location(remarkFormatFlag),
        llvm::cl::value_desc("format"),
        llvm::cl::init(RemarkFormat::REMARK_FORMAT_STDOUT),
        llvm::cl::values(clEnumValN(RemarkFormat::REMARK_FORMAT_STDOUT,
                                    "emitRemark",
                                    "Print as emitRemark to command-line"),
                         clEnumValN(RemarkFormat::REMARK_FORMAT_YAML, "yaml",
                                    "Print yaml file"),
                         clEnumValN(RemarkFormat::REMARK_FORMAT_BITSTREAM,
                                    "bitstream", "Print bitstream file")),
        llvm::cl::cat(remarkCategory)};

    static llvm::cl::opt<RemarkPolicy, /*ExternalStorage=*/true> remarkPolicy{
        "remark-policy",
        llvm::cl::desc("Specify the policy for remark output."),
        cl::location(remarkPolicyFlag),
        llvm::cl::value_desc("format"),
        llvm::cl::init(RemarkPolicy::REMARK_POLICY_ALL),
        llvm::cl::values(clEnumValN(RemarkPolicy::REMARK_POLICY_ALL, "all",
                                    "Print all remarks"),
                         clEnumValN(RemarkPolicy::REMARK_POLICY_FINAL, "final",
                                    "Print final remarks")),
        llvm::cl::cat(remarkCategory)};

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksAll(
        "remarks-filter",
        cl::desc("Show all remarks: passed, missed, failed, analysis"),
        cl::location(remarksAllFilterFlag), cl::init(""),
        cl::cat(remarkCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksFile(
        "remarks-output-file",
        cl::desc(
            "Output file for yaml and bitstream remark formats. Default is "
            "mlir-remarks.yaml or mlir-remarks.bitstream"),
        cl::location(remarksOutputFileFlag), cl::init(""),
        cl::cat(remarkCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksPassed(
        "remarks-filter-passed", cl::desc("Show passed remarks"),
        cl::location(remarksPassedFilterFlag), cl::init(""),
        cl::cat(remarkCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksFailed(
        "remarks-filter-failed", cl::desc("Show failed remarks"),
        cl::location(remarksFailedFilterFlag), cl::init(""),
        cl::cat(remarkCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksMissed(
        "remarks-filter-missed", cl::desc("Show missed remarks"),
        cl::location(remarksMissedFilterFlag), cl::init(""),
        cl::cat(remarkCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> remarksAnalyse(
        "remarks-filter-analyse", cl::desc("Show analysis remarks"),
        cl::location(remarksAnalyseFilterFlag), cl::init(""),
        cl::cat(remarkCategory));

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

/// A scoped diagnostic handler that suppresses certain diagnostics based on
/// the verbosity level and whether the diagnostic is a note.
class DiagnosticFilter : public ScopedDiagnosticHandler {
public:
  DiagnosticFilter(MLIRContext *ctx, VerbosityLevel verbosityLevel,
                   bool showNotes = true)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([verbosityLevel, showNotes](Diagnostic &diag) {
      auto severity = diag.getSeverity();
      switch (severity) {
      case mlir::DiagnosticSeverity::Error:
        // failure indicates that the error is not handled by the filter and
        // goes through to the default handler. Therefore, the error can be
        // successfully printed.
        return failure();
      case mlir::DiagnosticSeverity::Warning:
        if (verbosityLevel == VerbosityLevel::ErrorsOnly)
          return success();
        else
          return failure();
      case mlir::DiagnosticSeverity::Remark:
        if (verbosityLevel == VerbosityLevel::ErrorsOnly ||
            verbosityLevel == VerbosityLevel::ErrorsAndWarnings)
          return success();
        else
          return failure();
      case mlir::DiagnosticSeverity::Note:
        if (showNotes)
          return failure();
        else
          return success();
      }
      llvm_unreachable("Unknown diagnostic severity");
    });
  }
};
} // namespace

ManagedStatic<MlirOptMainConfigCLOptions> clOptionsConfig;

void MlirOptMainConfig::registerCLOptions(DialectRegistry &registry) {
  clOptionsConfig->setDialectPluginsCallback(registry);
  tracing::DebugConfig::registerCLOptions();
}

MlirOptMainConfig MlirOptMainConfig::createFromCLOptions() {
  clOptionsConfig->setDebugConfig(tracing::DebugConfig::createFromCLOptions());
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

LogicalResult loadIRDLDialects(StringRef irdlFile, MLIRContext &ctx) {
  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  ctx.appendDialectRegistry(registry);

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<MemoryBuffer> file = openInputFile(irdlFile, &errorMessage);
  if (!file) {
    emitError(UnknownLoc::get(&ctx)) << errorMessage;
    return failure();
  }

  // Give the buffer to the source manager.
  // This will be picked up by the parser.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &ctx);

  // Parse the input file.
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &ctx));
  if (!module)
    return failure();

  // Load IRDL dialects.
  return irdl::loadDialects(module.get());
}

// Return success if the module can correctly round-trip. This intended to test
// that the custom printers/parsers are complete.
static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirOptMainConfig &config,
                                       bool useBytecode) {
  // We use a new context to avoid resource handle renaming issue in the diff.
  MLIRContext roundtripContext;
  OwningOpRef<Operation *> roundtripModule;
  roundtripContext.appendDialectRegistry(
      op->getContext()->getDialectRegistry());
  if (op->getContext()->allowsUnregisteredDialects())
    roundtripContext.allowUnregisteredDialects();
  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, roundtripContext)))
    return failure();

  std::string testType = (useBytecode) ? "bytecode" : "textual";
  // Print a first time with custom format (or bytecode) and parse it back to
  // the roundtripModule.
  {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (useBytecode) {
      if (failed(writeBytecodeToFile(op, ostream))) {
        op->emitOpError()
            << "failed to write bytecode, cannot verify round-trip.\n";
        return failure();
      }
    } else {
      op->print(ostream,
                OpPrintingFlags().printGenericOpForm().enableDebugInfo());
    }
    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parseConfig(&roundtripContext, config.shouldVerifyOnParsing(),
                             &fallbackResourceMap);
    roundtripModule = parseSourceString<Operation *>(buffer, parseConfig);
    if (!roundtripModule) {
      op->emitOpError() << "failed to parse " << testType
                        << " content back, cannot verify round-trip.\n";
      return failure();
    }
  }

  // Print in the generic form for the reference module and the round-tripped
  // one and compare the outputs.
  std::string reference, roundtrip;
  {
    llvm::raw_string_ostream ostreamref(reference);
    op->print(ostreamref,
              OpPrintingFlags().printGenericOpForm().enableDebugInfo());
    llvm::raw_string_ostream ostreamrndtrip(roundtrip);
    roundtripModule.get()->print(
        ostreamrndtrip,
        OpPrintingFlags().printGenericOpForm().enableDebugInfo());
  }
  if (reference != roundtrip) {
    // TODO implement a diff.
    return op->emitOpError()
           << testType
           << " roundTrip testing roundtripped module differs "
              "from reference:\n<<<<<<Reference\n"
           << reference << "\n=====\n"
           << roundtrip << "\n>>>>>roundtripped\n";
  }

  return success();
}

static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirOptMainConfig &config) {
  auto txtStatus = doVerifyRoundTrip(op, config, /*useBytecode=*/false);
  auto bcStatus = doVerifyRoundTrip(op, config, /*useBytecode=*/true);
  return success(succeeded(txtStatus) && succeeded(bcStatus));
}

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
  ParserConfig parseConfig(context, config.shouldVerifyOnParsing(),
                           &fallbackResourceMap);
  if (config.shouldRunReproducer())
    reproOptions.attachResourceParser(parseConfig);

  // Parse the input file and reset the context threading state.
  TimingScope parserTiming = timing.nest("Parser");
  OwningOpRef<Operation *> op = parseSourceFileForTool(
      sourceMgr, parseConfig, !config.shouldUseExplicitModule());
  parserTiming.stop();
  if (!op)
    return failure();

  // Perform round-trip verification if requested
  if (config.shouldVerifyRoundtrip() &&
      failed(doVerifyRoundTrip(op.get(), config)))
    return failure();

  context->enableMultithreading(wasThreadingEnabled);
  // Set the remark categories and policy.
  remark::RemarkCategories cats{
      config.getRemarksAllFilter(), config.getRemarksPassedFilter(),
      config.getRemarksMissedFilter(), config.getRemarksAnalyseFilter(),
      config.getRemarksFailedFilter()};

  mlir::MLIRContext &ctx = *context;
  // Helper to create the appropriate policy based on configuration
  auto createPolicy = [&config]()
      -> std::unique_ptr<mlir::remark::detail::RemarkEmittingPolicyBase> {
    if (config.getRemarkPolicy() == RemarkPolicy::REMARK_POLICY_ALL)
      return std::make_unique<mlir::remark::RemarkEmittingPolicyAll>();
    if (config.getRemarkPolicy() == RemarkPolicy::REMARK_POLICY_FINAL)
      return std::make_unique<mlir::remark::RemarkEmittingPolicyFinal>();

    llvm_unreachable("Invalid remark policy");
  };

  switch (config.getRemarkFormat()) {
  case RemarkFormat::REMARK_FORMAT_STDOUT:
    if (failed(mlir::remark::enableOptimizationRemarks(
            ctx, nullptr, createPolicy(), cats, true /*printAsEmitRemarks*/)))
      return failure();
    break;

  case RemarkFormat::REMARK_FORMAT_YAML: {
    std::string file = config.getRemarksOutputFile().empty()
                           ? "mlir-remarks.yaml"
                           : config.getRemarksOutputFile();
    if (failed(mlir::remark::enableOptimizationRemarksWithLLVMStreamer(
            ctx, file, llvm::remarks::Format::YAML, createPolicy(), cats)))
      return failure();
    break;
  }

  case RemarkFormat::REMARK_FORMAT_BITSTREAM: {
    std::string file = config.getRemarksOutputFile().empty()
                           ? "mlir-remarks.bitstream"
                           : config.getRemarksOutputFile();
    if (failed(mlir::remark::enableOptimizationRemarksWithLLVMStreamer(
            ctx, file, llvm::remarks::Format::Bitstream, createPolicy(), cats)))
      return failure();
    break;
  }
  }

  // Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (config.shouldRunReproducer() && failed(reproOptions.apply(pm)))
    return failure();
  if (failed(config.setupPassPipeline(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();

  // Generate reproducers if requested
  if (!config.getReproducerFilename().empty()) {
    StringRef anchorName = pm.getAnyOpAnchorName();
    const auto &passes = pm.getPasses();
    makeReproducer(anchorName, passes, op.get(),
                   config.getReproducerFilename());
  }

  // Print the output.
  TimingScope outputTiming = timing.nest("Output");
  if (config.shouldEmitBytecode()) {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);
    if (auto v = config.bytecodeVersionToEmit())
      writerConfig.setDesiredBytecodeVersion(*v);
    if (config.shouldElideResourceDataFromBytecode())
      writerConfig.setElideResourceDataFlag();
    return writeBytecodeToFile(op.get(), os, writerConfig);
  }

  if (config.bytecodeVersionToEmit().has_value())
    return emitError(UnknownLoc::get(pm.getContext()))
           << "bytecode version while not emitting bytecode";
  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                    &fallbackResourceMap);
  os << OpWithState(op.get(), asmState) << '\n';

  // This is required if the remark policy is final. Otherwise, the remarks are
  // not emitted.
  if (remark::detail::RemarkEngine *engine = ctx.getRemarkEngine())
    engine->getRemarkEmittingPolicy()->finalize();

  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<MemoryBuffer> ownedBuffer,
              llvm::MemoryBufferRef sourceBuffer,
              const MlirOptMainConfig &config, DialectRegistry &registry,
              SourceMgrDiagnosticVerifierHandler *verifyHandler,
              llvm::ThreadPoolInterface *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  // Add the original buffer to the source manager to use for determining
  // locations.
  sourceMgr->AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(sourceBuffer,
                                       /*RequiresNullTerminator=*/false),
      SMLoc());
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Create a context just for the current buffer. Disable threading on
  // creation since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool)
    context.setThreadPool(*threadPool);
  if (verifyHandler)
    verifyHandler->registerInContext(&context);

  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, context)))
    return failure();

  // Parse the input file.
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  if (config.shouldVerifyDiagnostics())
    context.printOpOnDiagnostic(false);

  tracing::InstallDebugHandler installDebugHandler(context,
                                                   config.getDebugConfig());

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!config.shouldVerifyDiagnostics()) {
    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
    DiagnosticFilter diagnosticFilter(&context,
                                      config.getDiagnosticVerbosityLevel(),
                                      config.shouldShowNotes());
    return performActions(os, sourceMgr, &context, config);
  }

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, sourceMgr, &context, config);

  return success();
}

std::pair<std::string, std::string>
mlir::registerAndParseCLIOptions(int argc, char **argv,
                                 llvm::StringRef toolName,
                                 DialectRegistry &registry) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
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
  return std::make_pair(inputFilename.getValue(), outputFilename.getValue());
}

static LogicalResult printRegisteredDialects(DialectRegistry &registry) {
  llvm::outs() << "Available Dialects: ";
  interleave(registry.getDialectNames(), llvm::outs(), ",");
  llvm::outs() << "\n";
  return success();
}

static LogicalResult printRegisteredPassesAndReturn() {
  mlir::printRegisteredPasses();
  return success();
}

LogicalResult mlir::MlirOptMain(llvm::raw_ostream &outputStream,
                                std::unique_ptr<llvm::MemoryBuffer> buffer,
                                DialectRegistry &registry,
                                const MlirOptMainConfig &config) {
  if (config.shouldShowDialects())
    return printRegisteredDialects(registry);

  if (config.shouldListPasses())
    return printRegisteredPassesAndReturn();

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPoolInterface *threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(buffer->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/false),
      SMLoc());
  // Note: this creates a verifier handler independent of the the flag set, as
  // internally if the flag is not set, a new scoped diagnostic handler is
  // created which would intercept the diagnostics and verify them.
  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(
      sourceMgr, &threadPoolCtx, config.verifyDiagnosticsLevel());
  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     llvm::MemoryBufferRef sourceBuffer, raw_ostream &os) {
    return processBuffer(
        os, std::move(chunkBuffer), sourceBuffer, config, registry,
        config.shouldVerifyDiagnostics() ? &sourceMgrHandler : nullptr,
        threadPool);
  };
  LogicalResult status = splitAndProcessBuffer(
      llvm::MemoryBuffer::getMemBuffer(buffer->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/false),
      chunkFn, outputStream, config.inputSplitMarker(),
      config.outputSplitMarker());
  if (config.shouldVerifyDiagnostics() && failed(sourceMgrHandler.verify()))
    status = failure();
  return status;
}

LogicalResult mlir::MlirOptMain(int argc, char **argv,
                                llvm::StringRef inputFilename,
                                llvm::StringRef outputFilename,
                                DialectRegistry &registry) {

  InitLLVM y(argc, argv);

  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

  if (config.shouldShowDialects())
    return printRegisteredDialects(registry);

  if (config.shouldListPasses())
    return printRegisteredPassesAndReturn();

  // When reading from stdin and the input is a tty, it is often a user
  // mistake and the process "appears to be stuck". Print a message to let the
  // user know about it!
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

LogicalResult mlir::MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                                DialectRegistry &registry) {

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  return MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
}
