//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSMANAGER_H
#define MLIR_PASS_PASSMANAGER_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <vector>

namespace llvm {
class Any;
} // end namespace llvm

namespace mlir {
class AnalysisManager;
class Identifier;
class MLIRContext;
class Operation;
class Pass;
class PassInstrumentation;
class PassInstrumentor;

namespace detail {
struct OpPassManagerImpl;
class OpToOpPassAdaptor;
class PassCrashReproducerGenerator;
struct PassExecutionState;
} // end namespace detail

//===----------------------------------------------------------------------===//
// OpPassManager
//===----------------------------------------------------------------------===//

/// This class represents a pass manager that runs passes on a specific
/// operation type. This class is not constructed directly, but nested within
/// other OpPassManagers or the top-level PassManager.
class OpPassManager {
public:
  enum class Nesting { Implicit, Explicit };
  OpPassManager(Identifier name, Nesting nesting = Nesting::Explicit);
  OpPassManager(StringRef name, Nesting nesting = Nesting::Explicit);
  OpPassManager(OpPassManager &&rhs);
  OpPassManager(const OpPassManager &rhs);
  ~OpPassManager();
  OpPassManager &operator=(const OpPassManager &rhs);

  /// Iterator over the passes in this pass manager.
  using pass_iterator =
      llvm::pointee_iterator<MutableArrayRef<std::unique_ptr<Pass>>::iterator>;
  pass_iterator begin();
  pass_iterator end();
  iterator_range<pass_iterator> getPasses() { return {begin(), end()}; }

  using const_pass_iterator =
      llvm::pointee_iterator<ArrayRef<std::unique_ptr<Pass>>::const_iterator>;
  const_pass_iterator begin() const;
  const_pass_iterator end() const;
  iterator_range<const_pass_iterator> getPasses() const {
    return {begin(), end()};
  }

  /// Nest a new operation pass manager for the given operation kind under this
  /// pass manager.
  OpPassManager &nest(Identifier nestedName);
  OpPassManager &nest(StringRef nestedName);
  template <typename OpT> OpPassManager &nest() {
    return nest(OpT::getOperationName());
  }

  /// Add the given pass to this pass manager. If this pass has a concrete
  /// operation type, it must be the same type as this pass manager.
  void addPass(std::unique_ptr<Pass> pass);

  /// Add the given pass to a nested pass manager for the given operation kind
  /// `OpT`.
  template <typename OpT> void addNestedPass(std::unique_ptr<Pass> pass) {
    nest<OpT>().addPass(std::move(pass));
  }

  /// Returns the number of passes held by this manager.
  size_t size() const;

  /// Return the operation name that this pass manager operates on.
  Identifier getOpName(MLIRContext &context) const;

  /// Return the operation name that this pass manager operates on.
  StringRef getOpName() const;

  /// Returns the internal implementation instance.
  detail::OpPassManagerImpl &getImpl();

  /// Prints out the passes of the pass manager as the textual representation
  /// of pipelines.
  /// Note: The quality of the string representation depends entirely on the
  /// the correctness of per-pass overrides of Pass::printAsTextualPipeline.
  void printAsTextualPipeline(raw_ostream &os);

  /// Raw dump of the pass manager to llvm::errs().
  void dump();

  /// Merge the pass statistics of this class into 'other'.
  void mergeStatisticsInto(OpPassManager &other);

  /// Register dependent dialects for the current pass manager.
  /// This is forwarding to every pass in this PassManager, see the
  /// documentation for the same method on the Pass class.
  void getDependentDialects(DialectRegistry &dialects) const;

  /// Enable or disable the implicit nesting on this particular PassManager.
  /// This will also apply to any newly nested PassManager built from this
  /// instance.
  void setNesting(Nesting nesting);

  /// Return the current nesting mode.
  Nesting getNesting();

private:
  /// Initialize all of the passes within this pass manager with the given
  /// initialization generation. The initialization generation is used to detect
  /// if a pass manager has already been initialized.
  LogicalResult initialize(MLIRContext *context, unsigned newInitGeneration);

  /// A pointer to an internal implementation instance.
  std::unique_ptr<detail::OpPassManagerImpl> impl;

  /// Allow access to initialize.
  friend detail::OpToOpPassAdaptor;

  /// Allow access to the constructor.
  friend class PassManager;
  friend class Pass;

  /// Allow access.
  friend detail::OpPassManagerImpl;
};

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

/// An enum describing the different display modes for the information within
/// the pass manager.
enum class PassDisplayMode {
  // In this mode the results are displayed in a list sorted by total,
  // with each pass/analysis instance aggregated into one unique result.
  List,

  // In this mode the results are displayed in a nested pipeline view that
  // mirrors the internal pass pipeline that is being executed in the pass
  // manager.
  Pipeline,
};

/// The main pass manager and pipeline builder.
class PassManager : public OpPassManager {
public:
  /// Create a new pass manager under the given context with a specific nesting
  /// style. The created pass manager can schedule operations that match
  /// `operationName`.
  PassManager(MLIRContext *ctx, Nesting nesting = Nesting::Explicit,
              StringRef operationName = "module");
  PassManager(MLIRContext *ctx, StringRef operationName)
      : PassManager(ctx, Nesting::Explicit, operationName) {}
  ~PassManager();

  /// Run the passes within this manager on the provided operation. The
  /// specified operation must have the same name as the one provided the pass
  /// manager on construction.
  LogicalResult run(Operation *op);

  /// Return an instance of the context.
  MLIRContext *getContext() const { return context; }

  /// Enable support for the pass manager to generate a reproducer on the event
  /// of a crash or a pass failure. `outputFile` is a .mlir filename used to
  /// write the generated reproducer. If `genLocalReproducer` is true, the pass
  /// manager will attempt to generate a local reproducer that contains the
  /// smallest pipeline.
  void enableCrashReproducerGeneration(StringRef outputFile,
                                       bool genLocalReproducer = false);

  /// Streams on which to output crash reproducer.
  struct ReproducerStream {
    virtual ~ReproducerStream() = default;

    /// Description of the reproducer stream.
    virtual StringRef description() = 0;

    /// Stream on which to output reproducer.
    virtual raw_ostream &os() = 0;
  };

  /// Method type for constructing ReproducerStream.
  using ReproducerStreamFactory =
      std::function<std::unique_ptr<ReproducerStream>(std::string &error)>;

  /// Enable support for the pass manager to generate a reproducer on the event
  /// of a crash or a pass failure. `factory` is used to construct the streams
  /// to write the generated reproducer to. If `genLocalReproducer` is true, the
  /// pass manager will attempt to generate a local reproducer that contains the
  /// smallest pipeline.
  void enableCrashReproducerGeneration(ReproducerStreamFactory factory,
                                       bool genLocalReproducer = false);

  /// Runs the verifier after each individual pass.
  void enableVerifier(bool enabled = true);

  //===--------------------------------------------------------------------===//
  // Instrumentations
  //===--------------------------------------------------------------------===//

  /// Add the provided instrumentation to the pass manager.
  void addInstrumentation(std::unique_ptr<PassInstrumentation> pi);

  //===--------------------------------------------------------------------===//
  // IR Printing

  /// A configuration struct provided to the IR printer instrumentation.
  class IRPrinterConfig {
  public:
    using PrintCallbackFn = function_ref<void(raw_ostream &)>;

    /// Initialize the configuration.
    /// * 'printModuleScope' signals if the top-level module IR should always be
    ///   printed. This should only be set to true when multi-threading is
    ///   disabled, otherwise we may try to print IR that is being modified
    ///   asynchronously.
    /// * 'printAfterOnlyOnChange' signals that when printing the IR after a
    ///   pass, in the case of a non-failure, we should first check if any
    ///   potential mutations were made. This allows for reducing the number of
    ///   logs that don't contain meaningful changes.
    /// * 'printAfterOnlyOnFailure' signals that when printing the IR after a
    ///   pass, we only print in the case of a failure.
    ///     - This option should *not* be used with the other `printAfter` flags
    ///       above.
    /// * 'opPrintingFlags' sets up the printing flags to use when printing the
    ///   IR.
    explicit IRPrinterConfig(
        bool printModuleScope = false, bool printAfterOnlyOnChange = false,
        bool printAfterOnlyOnFailure = false,
        OpPrintingFlags opPrintingFlags = OpPrintingFlags());
    virtual ~IRPrinterConfig();

    /// A hook that may be overridden by a derived config that checks if the IR
    /// of 'operation' should be dumped *before* the pass 'pass' has been
    /// executed. If the IR should be dumped, 'printCallback' should be invoked
    /// with the stream to dump into.
    virtual void printBeforeIfEnabled(Pass *pass, Operation *operation,
                                      PrintCallbackFn printCallback);

    /// A hook that may be overridden by a derived config that checks if the IR
    /// of 'operation' should be dumped *after* the pass 'pass' has been
    /// executed. If the IR should be dumped, 'printCallback' should be invoked
    /// with the stream to dump into.
    virtual void printAfterIfEnabled(Pass *pass, Operation *operation,
                                     PrintCallbackFn printCallback);

    /// Returns true if the IR should always be printed at the top-level scope.
    bool shouldPrintAtModuleScope() const { return printModuleScope; }

    /// Returns true if the IR should only printed after a pass if the IR
    /// "changed".
    bool shouldPrintAfterOnlyOnChange() const { return printAfterOnlyOnChange; }

    /// Returns true if the IR should only printed after a pass if the pass
    /// "failed".
    bool shouldPrintAfterOnlyOnFailure() const {
      return printAfterOnlyOnFailure;
    }

    /// Returns the printing flags to be used to print the IR.
    OpPrintingFlags getOpPrintingFlags() const { return opPrintingFlags; }

  private:
    /// A flag that indicates if the IR should be printed at module scope.
    bool printModuleScope;

    /// A flag that indicates that the IR after a pass should only be printed if
    /// a change is detected.
    bool printAfterOnlyOnChange;

    /// A flag that indicates that the IR after a pass should only be printed if
    /// the pass failed.
    bool printAfterOnlyOnFailure;

    /// Flags to control printing behavior.
    OpPrintingFlags opPrintingFlags;
  };

  /// Add an instrumentation to print the IR before and after pass execution,
  /// using the provided configuration.
  void enableIRPrinting(std::unique_ptr<IRPrinterConfig> config);

  /// Add an instrumentation to print the IR before and after pass execution,
  /// using the provided fields to generate a default configuration:
  /// * 'shouldPrintBeforePass' and 'shouldPrintAfterPass' correspond to filter
  ///   functions that take a 'Pass *' and `Operation *`. These function should
  ///   return true if the IR should be printed or not.
  /// * 'printModuleScope' signals if the module IR should be printed, even
  ///   for non module passes.
  /// * 'printAfterOnlyOnChange' signals that when printing the IR after a
  ///   pass, in the case of a non-failure, we should first check if any
  ///   potential mutations were made.
  /// * 'printAfterOnlyOnFailure' signals that when printing the IR after a
  ///   pass, we only print in the case of a failure.
  ///     - This option should *not* be used with the other `printAfter` flags
  ///       above.
  /// * 'out' corresponds to the stream to output the printed IR to.
  /// * 'opPrintingFlags' sets up the printing flags to use when printing the
  ///   IR.
  void enableIRPrinting(
      std::function<bool(Pass *, Operation *)> shouldPrintBeforePass =
          [](Pass *, Operation *) { return true; },
      std::function<bool(Pass *, Operation *)> shouldPrintAfterPass =
          [](Pass *, Operation *) { return true; },
      bool printModuleScope = true, bool printAfterOnlyOnChange = true,
      bool printAfterOnlyOnFailure = false, raw_ostream &out = llvm::errs(),
      OpPrintingFlags opPrintingFlags = OpPrintingFlags());

  //===--------------------------------------------------------------------===//
  // Pass Timing

  /// Add an instrumentation to time the execution of passes and the computation
  /// of analyses. Timing will be reported by nesting timers into the provided
  /// `timingScope`.
  ///
  /// Note: Timing should be enabled after all other instrumentations to avoid
  /// any potential "ghost" timing from other instrumentations being
  /// unintentionally included in the timing results.
  void enableTiming(TimingScope &timingScope);

  /// Add an instrumentation to time the execution of passes and the computation
  /// of analyses. The pass manager will take ownership of the timing manager
  /// passed to the function and timing will be reported by nesting timers into
  /// the timing manager's root scope.
  ///
  /// Note: Timing should be enabled after all other instrumentations to avoid
  /// any potential "ghost" timing from other instrumentations being
  /// unintentionally included in the timing results.
  void enableTiming(std::unique_ptr<TimingManager> tm);

  /// Add an instrumentation to time the execution of passes and the computation
  /// of analyses. Creates a temporary TimingManager owned by this PassManager
  /// which will be used to report timing.
  ///
  /// Note: Timing should be enabled after all other instrumentations to avoid
  /// any potential "ghost" timing from other instrumentations being
  /// unintentionally included in the timing results.
  void enableTiming();

  //===--------------------------------------------------------------------===//
  // Pass Statistics

  /// Prompts the pass manager to print the statistics collected for each of the
  /// held passes after each call to 'run'.
  void
  enableStatistics(PassDisplayMode displayMode = PassDisplayMode::Pipeline);

private:
  /// Dump the statistics of the passes within this pass manager.
  void dumpStatistics();

  /// Run the pass manager with crash recovery enabled.
  LogicalResult runWithCrashRecovery(Operation *op, AnalysisManager am);

  /// Run the passes of the pass manager, and return the result.
  LogicalResult runPasses(Operation *op, AnalysisManager am);

  /// Context this PassManager was initialized with.
  MLIRContext *context;

  /// Flag that specifies if pass statistics should be dumped.
  Optional<PassDisplayMode> passStatisticsMode;

  /// A manager for pass instrumentations.
  std::unique_ptr<PassInstrumentor> instrumentor;

  /// An optional crash reproducer generator, if this pass manager is setup to
  /// generate reproducers.
  std::unique_ptr<detail::PassCrashReproducerGenerator> crashReproGenerator;

  /// A hash key used to detect when reinitialization is necessary.
  llvm::hash_code initializationKey;

  /// Flag that specifies if pass timing is enabled.
  bool passTiming : 1;

  /// A flag that indicates if the IR should be verified in between passes.
  bool verifyPasses : 1;
};

/// Register a set of useful command-line options that can be used to configure
/// a pass manager. The values of these options can be applied via the
/// 'applyPassManagerCLOptions' method below.
void registerPassManagerCLOptions();

/// Apply any values provided to the pass manager options that were registered
/// with 'registerPassManagerOptions'.
void applyPassManagerCLOptions(PassManager &pm);

/// Apply any values provided to the timing manager options that were registered
/// with `registerDefaultTimingManagerOptions`. This is a handy helper function
/// if you do not want to bother creating your own timing manager and passing it
/// to the pass manager.
void applyDefaultTimingPassManagerCLOptions(PassManager &pm);

} // end namespace mlir

#endif // MLIR_PASS_PASSMANAGER_H
