//===- Pass.cpp - Pass infrastructure implementation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common pass infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "PassDetail.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Attempt to initialize the options of this pass from the given string.
LogicalResult Pass::initializeOptions(StringRef options) {
  return passOptions.parseFromString(options);
}

/// Copy the option values from 'other', which is another instance of this
/// pass.
void Pass::copyOptionValuesFrom(const Pass *other) {
  passOptions.copyOptionValuesFrom(other->passOptions);
}

/// Prints out the pass in the textual representation of pipelines. If this is
/// an adaptor pass, print with the op_name(sub_pass,...) format.
void Pass::printAsTextualPipeline(raw_ostream &os) {
  // Special case for adaptors to use the 'op_name(sub_passes)' format.
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(this)) {
    llvm::interleaveComma(adaptor->getPassManagers(), os,
                          [&](OpPassManager &pm) {
                            os << pm.getOpName() << "(";
                            pm.printAsTextualPipeline(os);
                            os << ")";
                          });
    return;
  }
  // Otherwise, print the pass argument followed by its options. If the pass
  // doesn't have an argument, print the name of the pass to give some indicator
  // of what pass was run.
  StringRef argument = getArgument();
  if (!argument.empty())
    os << argument;
  else
    os << "unknown<" << getName() << ">";
  passOptions.print(os);
}

//===----------------------------------------------------------------------===//
// OpPassManagerImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct OpPassManagerImpl {
  OpPassManagerImpl(Identifier identifier, OpPassManager::Nesting nesting)
      : name(identifier.str()), identifier(identifier), nesting(nesting) {}
  OpPassManagerImpl(StringRef name, OpPassManager::Nesting nesting)
      : name(name), nesting(nesting) {}

  /// Merge the passes of this pass manager into the one provided.
  void mergeInto(OpPassManagerImpl &rhs);

  /// Nest a new operation pass manager for the given operation kind under this
  /// pass manager.
  OpPassManager &nest(Identifier nestedName);
  OpPassManager &nest(StringRef nestedName);

  /// Add the given pass to this pass manager. If this pass has a concrete
  /// operation type, it must be the same type as this pass manager.
  void addPass(std::unique_ptr<Pass> pass);

  /// Coalesce adjacent AdaptorPasses into one large adaptor. This runs
  /// recursively through the pipeline graph.
  void coalesceAdjacentAdaptorPasses();

  /// Split all of AdaptorPasses such that each adaptor only contains one leaf
  /// pass.
  void splitAdaptorPasses();

  Identifier getOpName(MLIRContext &context) {
    if (!identifier)
      identifier = Identifier::get(name, &context);
    return *identifier;
  }

  /// The name of the operation that passes of this pass manager operate on.
  std::string name;

  /// The cached identifier (internalized in the context) for the name of the
  /// operation that passes of this pass manager operate on.
  Optional<Identifier> identifier;

  /// The set of passes to run as part of this pass manager.
  std::vector<std::unique_ptr<Pass>> passes;

  /// Control the implicit nesting of passes that mismatch the name set for this
  /// OpPassManager.
  OpPassManager::Nesting nesting;
};
} // end namespace detail
} // end namespace mlir

void OpPassManagerImpl::mergeInto(OpPassManagerImpl &rhs) {
  assert(name == rhs.name && "merging unrelated pass managers");
  for (auto &pass : passes)
    rhs.passes.push_back(std::move(pass));
  passes.clear();
}

OpPassManager &OpPassManagerImpl::nest(Identifier nestedName) {
  OpPassManager nested(nestedName, nesting);
  auto *adaptor = new OpToOpPassAdaptor(std::move(nested));
  addPass(std::unique_ptr<Pass>(adaptor));
  return adaptor->getPassManagers().front();
}

OpPassManager &OpPassManagerImpl::nest(StringRef nestedName) {
  OpPassManager nested(nestedName, nesting);
  auto *adaptor = new OpToOpPassAdaptor(std::move(nested));
  addPass(std::unique_ptr<Pass>(adaptor));
  return adaptor->getPassManagers().front();
}

void OpPassManagerImpl::addPass(std::unique_ptr<Pass> pass) {
  // If this pass runs on a different operation than this pass manager, then
  // implicitly nest a pass manager for this operation if enabled.
  auto passOpName = pass->getOpName();
  if (passOpName && passOpName->str() != name) {
    if (nesting == OpPassManager::Nesting::Implicit)
      return nest(*passOpName).addPass(std::move(pass));
    llvm::report_fatal_error(llvm::Twine("Can't add pass '") + pass->getName() +
                             "' restricted to '" + *passOpName +
                             "' on a PassManager intended to run on '" + name +
                             "', did you intend to nest?");
  }

  passes.emplace_back(std::move(pass));
}

void OpPassManagerImpl::coalesceAdjacentAdaptorPasses() {
  // Bail out early if there are no adaptor passes.
  if (llvm::none_of(passes, [](std::unique_ptr<Pass> &pass) {
        return isa<OpToOpPassAdaptor>(pass.get());
      }))
    return;

  // Walk the pass list and merge adjacent adaptors.
  OpToOpPassAdaptor *lastAdaptor = nullptr;
  for (auto it = passes.begin(), e = passes.end(); it != e; ++it) {
    // Check to see if this pass is an adaptor.
    if (auto *currentAdaptor = dyn_cast<OpToOpPassAdaptor>(it->get())) {
      // If it is the first adaptor in a possible chain, remember it and
      // continue.
      if (!lastAdaptor) {
        lastAdaptor = currentAdaptor;
        continue;
      }

      // Otherwise, merge into the existing adaptor and delete the current one.
      currentAdaptor->mergeInto(*lastAdaptor);
      it->reset();
    } else if (lastAdaptor) {
      // If this pass is not an adaptor, then coalesce and forget any existing
      // adaptor.
      for (auto &pm : lastAdaptor->getPassManagers())
        pm.getImpl().coalesceAdjacentAdaptorPasses();
      lastAdaptor = nullptr;
    }
  }

  // If there was an adaptor at the end of the manager, coalesce it as well.
  if (lastAdaptor) {
    for (auto &pm : lastAdaptor->getPassManagers())
      pm.getImpl().coalesceAdjacentAdaptorPasses();
  }

  // Now that the adaptors have been merged, erase the empty slot corresponding
  // to the merged adaptors that were nulled-out in the loop above.
  llvm::erase_if(passes, std::logical_not<std::unique_ptr<Pass>>());
}

void OpPassManagerImpl::splitAdaptorPasses() {
  std::vector<std::unique_ptr<Pass>> oldPasses;
  std::swap(passes, oldPasses);

  for (std::unique_ptr<Pass> &pass : oldPasses) {
    // If this pass isn't an adaptor, move it directly to the new pass list.
    auto *currentAdaptor = dyn_cast<OpToOpPassAdaptor>(pass.get());
    if (!currentAdaptor) {
      addPass(std::move(pass));
      continue;
    }

    // Otherwise, split the adaptors of each manager within the adaptor.
    for (OpPassManager &adaptorPM : currentAdaptor->getPassManagers()) {
      adaptorPM.getImpl().splitAdaptorPasses();
      for (std::unique_ptr<Pass> &nestedPass : adaptorPM.getImpl().passes)
        nest(adaptorPM.getOpName()).addPass(std::move(nestedPass));
    }
  }
}

//===----------------------------------------------------------------------===//
// OpPassManager
//===----------------------------------------------------------------------===//

OpPassManager::OpPassManager(Identifier name, Nesting nesting)
    : impl(new OpPassManagerImpl(name, nesting)) {}
OpPassManager::OpPassManager(StringRef name, Nesting nesting)
    : impl(new OpPassManagerImpl(name, nesting)) {}
OpPassManager::OpPassManager(OpPassManager &&rhs) : impl(std::move(rhs.impl)) {}
OpPassManager::OpPassManager(const OpPassManager &rhs) { *this = rhs; }
OpPassManager &OpPassManager::operator=(const OpPassManager &rhs) {
  impl.reset(new OpPassManagerImpl(rhs.impl->name, rhs.impl->nesting));
  for (auto &pass : rhs.impl->passes)
    impl->passes.emplace_back(pass->clone());
  return *this;
}

OpPassManager::~OpPassManager() {}

OpPassManager::pass_iterator OpPassManager::begin() {
  return MutableArrayRef<std::unique_ptr<Pass>>{impl->passes}.begin();
}
OpPassManager::pass_iterator OpPassManager::end() {
  return MutableArrayRef<std::unique_ptr<Pass>>{impl->passes}.end();
}

OpPassManager::const_pass_iterator OpPassManager::begin() const {
  return ArrayRef<std::unique_ptr<Pass>>{impl->passes}.begin();
}
OpPassManager::const_pass_iterator OpPassManager::end() const {
  return ArrayRef<std::unique_ptr<Pass>>{impl->passes}.end();
}

/// Nest a new operation pass manager for the given operation kind under this
/// pass manager.
OpPassManager &OpPassManager::nest(Identifier nestedName) {
  return impl->nest(nestedName);
}
OpPassManager &OpPassManager::nest(StringRef nestedName) {
  return impl->nest(nestedName);
}

/// Add the given pass to this pass manager. If this pass has a concrete
/// operation type, it must be the same type as this pass manager.
void OpPassManager::addPass(std::unique_ptr<Pass> pass) {
  impl->addPass(std::move(pass));
}

/// Returns the number of passes held by this manager.
size_t OpPassManager::size() const { return impl->passes.size(); }

/// Returns the internal implementation instance.
OpPassManagerImpl &OpPassManager::getImpl() { return *impl; }

/// Return the operation name that this pass manager operates on.
StringRef OpPassManager::getOpName() const { return impl->name; }

/// Return the operation name that this pass manager operates on.
Identifier OpPassManager::getOpName(MLIRContext &context) const {
  return impl->getOpName(context);
}

/// Prints out the given passes as the textual representation of a pipeline.
static void printAsTextualPipeline(ArrayRef<std::unique_ptr<Pass>> passes,
                                   raw_ostream &os) {
  llvm::interleaveComma(passes, os, [&](const std::unique_ptr<Pass> &pass) {
    pass->printAsTextualPipeline(os);
  });
}

/// Prints out the passes of the pass manager as the textual representation
/// of pipelines.
void OpPassManager::printAsTextualPipeline(raw_ostream &os) {
  ::printAsTextualPipeline(impl->passes, os);
}

void OpPassManager::dump() {
  llvm::errs() << "Pass Manager with " << impl->passes.size() << " passes: ";
  ::printAsTextualPipeline(impl->passes, llvm::errs());
  llvm::errs() << "\n";
}

static void registerDialectsForPipeline(const OpPassManager &pm,
                                        DialectRegistry &dialects) {
  for (const Pass &pass : pm.getPasses())
    pass.getDependentDialects(dialects);
}

void OpPassManager::getDependentDialects(DialectRegistry &dialects) const {
  registerDialectsForPipeline(*this, dialects);
}

OpPassManager::Nesting OpPassManager::getNesting() { return impl->nesting; }

void OpPassManager::setNesting(Nesting nesting) { impl->nesting = nesting; }

//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

LogicalResult OpToOpPassAdaptor::run(Pass *pass, Operation *op,
                                     AnalysisManager am, bool verifyPasses) {
  if (!op->getName().getAbstractOperation())
    return op->emitOpError()
           << "trying to schedule a pass on an unregistered operation";
  if (!op->getName().getAbstractOperation()->hasProperty(
          OperationProperty::IsolatedFromAbove))
    return op->emitOpError() << "trying to schedule a pass on an operation not "
                                "marked as 'IsolatedFromAbove'";

  // Initialize the pass state with a callback for the pass to dynamically
  // execute a pipeline on the currently visited operation.
  auto dynamic_pipeline_callback =
      [op, &am, verifyPasses](OpPassManager &pipeline,
                              Operation *root) -> LogicalResult {
    if (!op->isAncestor(root))
      return root->emitOpError()
             << "Trying to schedule a dynamic pipeline on an "
                "operation that isn't "
                "nested under the current operation the pass is processing";

    AnalysisManager nestedAm = am.nest(root);
    return OpToOpPassAdaptor::runPipeline(pipeline.getPasses(), root, nestedAm,
                                          verifyPasses);
  };
  pass->passState.emplace(op, am, dynamic_pipeline_callback);
  // Instrument before the pass has run.
  PassInstrumentor *pi = am.getPassInstrumentor();
  if (pi)
    pi->runBeforePass(pass, op);

  // Invoke the virtual runOnOperation method.
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(pass))
    adaptor->runOnOperation(verifyPasses);
  else
    pass->runOnOperation();
  bool passFailed = pass->passState->irAndPassFailed.getInt();

  // Invalidate any non preserved analyses.
  am.invalidate(pass->passState->preservedAnalyses);

  // Run the verifier if this pass didn't fail already.
  if (!passFailed && verifyPasses)
    passFailed = failed(verify(op));

  // Instrument after the pass has run.
  if (pi) {
    if (passFailed)
      pi->runAfterPassFailed(pass, op);
    else
      pi->runAfterPass(pass, op);
  }

  // Return if the pass signaled a failure.
  return failure(passFailed);
}

/// Run the given operation and analysis manager on a provided op pass manager.
LogicalResult OpToOpPassAdaptor::runPipeline(
    iterator_range<OpPassManager::pass_iterator> passes, Operation *op,
    AnalysisManager am, bool verifyPasses) {
  auto scope_exit = llvm::make_scope_exit([&] {
    // Clear out any computed operation analyses. These analyses won't be used
    // any more in this pipeline, and this helps reduce the current working set
    // of memory. If preserving these analyses becomes important in the future
    // we can re-evaluate this.
    am.clear();
  });

  // Run the pipeline over the provided operation.
  for (Pass &pass : passes)
    if (failed(run(&pass, op, am, verifyPasses)))
      return failure();

  return success();
}

/// Find an operation pass manager that can operate on an operation of the given
/// type, or nullptr if one does not exist.
static OpPassManager *findPassManagerFor(MutableArrayRef<OpPassManager> mgrs,
                                         StringRef name) {
  auto it = llvm::find_if(
      mgrs, [&](OpPassManager &mgr) { return mgr.getOpName() == name; });
  return it == mgrs.end() ? nullptr : &*it;
}

/// Find an operation pass manager that can operate on an operation of the given
/// type, or nullptr if one does not exist.
static OpPassManager *findPassManagerFor(MutableArrayRef<OpPassManager> mgrs,
                                         Identifier name,
                                         MLIRContext &context) {
  auto it = llvm::find_if(
      mgrs, [&](OpPassManager &mgr) { return mgr.getOpName(context) == name; });
  return it == mgrs.end() ? nullptr : &*it;
}

OpToOpPassAdaptor::OpToOpPassAdaptor(OpPassManager &&mgr) {
  mgrs.emplace_back(std::move(mgr));
}

void OpToOpPassAdaptor::getDependentDialects(DialectRegistry &dialects) const {
  for (auto &pm : mgrs)
    pm.getDependentDialects(dialects);
}

/// Merge the current pass adaptor into given 'rhs'.
void OpToOpPassAdaptor::mergeInto(OpToOpPassAdaptor &rhs) {
  for (auto &pm : mgrs) {
    // If an existing pass manager exists, then merge the given pass manager
    // into it.
    if (auto *existingPM = findPassManagerFor(rhs.mgrs, pm.getOpName())) {
      pm.getImpl().mergeInto(existingPM->getImpl());
    } else {
      // Otherwise, add the given pass manager to the list.
      rhs.mgrs.emplace_back(std::move(pm));
    }
  }
  mgrs.clear();

  // After coalescing, sort the pass managers within rhs by name.
  llvm::array_pod_sort(rhs.mgrs.begin(), rhs.mgrs.end(),
                       [](const OpPassManager *lhs, const OpPassManager *rhs) {
                         return lhs->getOpName().compare(rhs->getOpName());
                       });
}

/// Returns the adaptor pass name.
std::string OpToOpPassAdaptor::getAdaptorName() {
  std::string name = "Pipeline Collection : [";
  llvm::raw_string_ostream os(name);
  llvm::interleaveComma(getPassManagers(), os, [&](OpPassManager &pm) {
    os << '\'' << pm.getOpName() << '\'';
  });
  os << ']';
  return os.str();
}

void OpToOpPassAdaptor::runOnOperation() {
  llvm_unreachable(
      "Unexpected call to Pass::runOnOperation() on OpToOpPassAdaptor");
}

/// Run the held pipeline over all nested operations.
void OpToOpPassAdaptor::runOnOperation(bool verifyPasses) {
  if (getContext().isMultithreadingEnabled())
    runOnOperationAsyncImpl(verifyPasses);
  else
    runOnOperationImpl(verifyPasses);
}

/// Run this pass adaptor synchronously.
void OpToOpPassAdaptor::runOnOperationImpl(bool verifyPasses) {
  auto am = getAnalysisManager();
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        this};
  auto *instrumentor = am.getPassInstrumentor();
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        auto *mgr = findPassManagerFor(mgrs, op.getName().getIdentifier(),
                                       *op.getContext());
        if (!mgr)
          continue;
        Identifier opName = mgr->getOpName(*getOperation()->getContext());

        // Run the held pipeline over the current operation.
        if (instrumentor)
          instrumentor->runBeforePipeline(opName, parentInfo);
        LogicalResult result =
            runPipeline(mgr->getPasses(), &op, am.nest(&op), verifyPasses);
        if (instrumentor)
          instrumentor->runAfterPipeline(opName, parentInfo);

        if (failed(result))
          return signalPassFailure();
      }
    }
  }
}

/// Utility functor that checks if the two ranges of pass managers have a size
/// mismatch.
static bool hasSizeMismatch(ArrayRef<OpPassManager> lhs,
                            ArrayRef<OpPassManager> rhs) {
  return lhs.size() != rhs.size() ||
         llvm::any_of(llvm::seq<size_t>(0, lhs.size()),
                      [&](size_t i) { return lhs[i].size() != rhs[i].size(); });
}

/// Run this pass adaptor synchronously.
void OpToOpPassAdaptor::runOnOperationAsyncImpl(bool verifyPasses) {
  AnalysisManager am = getAnalysisManager();

  // Create the async executors if they haven't been created, or if the main
  // pipeline has changed.
  if (asyncExecutors.empty() || hasSizeMismatch(asyncExecutors.front(), mgrs))
    asyncExecutors.assign(llvm::hardware_concurrency().compute_thread_count(),
                          mgrs);

  // Run a prepass over the module to collect the operations to execute over.
  // This ensures that an analysis manager exists for each operation, as well as
  // providing a queue of operations to execute over.
  std::vector<std::pair<Operation *, AnalysisManager>> opAMPairs;
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        // Add this operation iff the name matches any of the pass managers.
        if (findPassManagerFor(mgrs, op.getName().getIdentifier(),
                               getContext()))
          opAMPairs.emplace_back(&op, am.nest(&op));
      }
    }
  }

  // A parallel diagnostic handler that provides deterministic diagnostic
  // ordering.
  ParallelDiagnosticHandler diagHandler(&getContext());

  // An index for the current operation/analysis manager pair.
  std::atomic<unsigned> opIt(0);

  // Get the current thread for this adaptor.
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        this};
  auto *instrumentor = am.getPassInstrumentor();

  // An atomic failure variable for the async executors.
  std::atomic<bool> passFailed(false);
  llvm::parallelForEach(
      asyncExecutors.begin(),
      std::next(asyncExecutors.begin(),
                std::min(asyncExecutors.size(), opAMPairs.size())),
      [&](MutableArrayRef<OpPassManager> pms) {
        for (auto e = opAMPairs.size(); !passFailed && opIt < e;) {
          // Get the next available operation index.
          unsigned nextID = opIt++;
          if (nextID >= e)
            break;

          // Set the order id for this thread in the diagnostic handler.
          diagHandler.setOrderIDForThread(nextID);

          // Get the pass manager for this operation and execute it.
          auto &it = opAMPairs[nextID];
          auto *pm = findPassManagerFor(
              pms, it.first->getName().getIdentifier(), getContext());
          assert(pm && "expected valid pass manager for operation");

          Identifier opName = pm->getOpName(*getOperation()->getContext());
          if (instrumentor)
            instrumentor->runBeforePipeline(opName, parentInfo);
          auto pipelineResult =
              runPipeline(pm->getPasses(), it.first, it.second, verifyPasses);
          if (instrumentor)
            instrumentor->runAfterPipeline(opName, parentInfo);

          // Drop this thread from being tracked by the diagnostic handler.
          // After this task has finished, the thread may be used outside of
          // this pass manager context meaning that we don't want to track
          // diagnostics from it anymore.
          diagHandler.eraseOrderIDForThread();

          // Handle a failed pipeline result.
          if (failed(pipelineResult)) {
            passFailed = true;
            break;
          }
        }
      });

  // Signal a failure if any of the executors failed.
  if (passFailed)
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// PassCrashReproducer
//===----------------------------------------------------------------------===//

namespace {
/// This class contains all of the context for generating a recovery reproducer.
/// Each recovery context is registered globally to allow for generating
/// reproducers when a signal is raised, such as a segfault.
struct RecoveryReproducerContext {
  RecoveryReproducerContext(MutableArrayRef<std::unique_ptr<Pass>> passes,
                            ModuleOp module, StringRef filename,
                            bool disableThreads, bool verifyPasses);
  ~RecoveryReproducerContext();

  /// Generate a reproducer with the current context.
  LogicalResult generate(std::string &error);

private:
  /// This function is invoked in the event of a crash.
  static void crashHandler(void *);

  /// Register a signal handler to run in the event of a crash.
  static void registerSignalHandler();

  /// The textual description of the currently executing pipeline.
  std::string pipeline;

  /// The MLIR module representing the IR before the crash.
  OwningModuleRef module;

  /// The filename to use when generating the reproducer.
  StringRef filename;

  /// Various pass manager and context flags.
  bool disableThreads;
  bool verifyPasses;

  /// The current set of active reproducer contexts. This is used in the event
  /// of a crash. This is not thread_local as the pass manager may produce any
  /// number of child threads. This uses a set to allow for multiple MLIR pass
  /// managers to be running at the same time.
  static llvm::ManagedStatic<llvm::sys::SmartMutex<true>> reproducerMutex;
  static llvm::ManagedStatic<
      llvm::SmallSetVector<RecoveryReproducerContext *, 1>>
      reproducerSet;
};
} // end anonymous namespace

llvm::ManagedStatic<llvm::sys::SmartMutex<true>>
    RecoveryReproducerContext::reproducerMutex;
llvm::ManagedStatic<llvm::SmallSetVector<RecoveryReproducerContext *, 1>>
    RecoveryReproducerContext::reproducerSet;

RecoveryReproducerContext::RecoveryReproducerContext(
    MutableArrayRef<std::unique_ptr<Pass>> passes, ModuleOp module,
    StringRef filename, bool disableThreads, bool verifyPasses)
    : module(module.clone()), filename(filename),
      disableThreads(disableThreads), verifyPasses(verifyPasses) {
  // Grab the textual pipeline being executed..
  {
    llvm::raw_string_ostream pipelineOS(pipeline);
    ::printAsTextualPipeline(passes, pipelineOS);
  }

  // Make sure that the handler is registered, and update the current context.
  llvm::sys::SmartScopedLock<true> producerLock(*reproducerMutex);
  if (reproducerSet->empty())
    llvm::CrashRecoveryContext::Enable();
  registerSignalHandler();
  reproducerSet->insert(this);
}

RecoveryReproducerContext::~RecoveryReproducerContext() {
  llvm::sys::SmartScopedLock<true> producerLock(*reproducerMutex);
  reproducerSet->remove(this);
  if (reproducerSet->empty())
    llvm::CrashRecoveryContext::Disable();
}

LogicalResult RecoveryReproducerContext::generate(std::string &error) {
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(filename, &error);
  if (!outputFile)
    return failure();
  auto &outputOS = outputFile->os();

  // Output the current pass manager configuration.
  outputOS << "// configuration: -pass-pipeline='" << pipeline << "'";
  if (disableThreads)
    outputOS << " -mlir-disable-threading";

  // TODO: Should this also be configured with a pass manager flag?
  outputOS << "\n// note: verifyPasses=" << (verifyPasses ? "true" : "false")
           << "\n";

  // Output the .mlir module.
  module->print(outputOS);
  outputFile->keep();
  return success();
}

void RecoveryReproducerContext::crashHandler(void *) {
  // Walk the current stack of contexts and generate a reproducer for each one.
  // We can't know for certain which one was the cause, so we need to generate
  // a reproducer for all of them.
  std::string ignored;
  for (RecoveryReproducerContext *context : *reproducerSet)
    context->generate(ignored);
}

void RecoveryReproducerContext::registerSignalHandler() {
  // Ensure that the handler is only registered once.
  static bool registered =
      (llvm::sys::AddSignalHandler(crashHandler, nullptr), false);
  (void)registered;
}

/// Run the pass manager with crash recover enabled.
LogicalResult PassManager::runWithCrashRecovery(ModuleOp module,
                                                AnalysisManager am) {
  // If this isn't a local producer, run all of the passes in recovery mode.
  if (!localReproducer)
    return runWithCrashRecovery(impl->passes, module, am);

  // Split the passes within adaptors to ensure that each pass can be run in
  // isolation.
  impl->splitAdaptorPasses();

  // If this is a local producer, run each of the passes individually.
  MutableArrayRef<std::unique_ptr<Pass>> passes = impl->passes;
  for (std::unique_ptr<Pass> &pass : passes)
    if (failed(runWithCrashRecovery(pass, module, am)))
      return failure();
  return success();
}

/// Run the given passes with crash recover enabled.
LogicalResult
PassManager::runWithCrashRecovery(MutableArrayRef<std::unique_ptr<Pass>> passes,
                                  ModuleOp module, AnalysisManager am) {
  RecoveryReproducerContext context(passes, module, *crashReproducerFileName,
                                    !getContext()->isMultithreadingEnabled(),
                                    verifyPasses);

  // Safely invoke the passes within a recovery context.
  LogicalResult passManagerResult = failure();
  llvm::CrashRecoveryContext recoveryContext;
  recoveryContext.RunSafelyOnThread([&] {
    for (std::unique_ptr<Pass> &pass : passes)
      if (failed(OpToOpPassAdaptor::run(pass.get(), module, am, verifyPasses)))
        return;
    passManagerResult = success();
  });
  if (succeeded(passManagerResult))
    return success();

  std::string error;
  if (failed(context.generate(error)))
    return module.emitError("<MLIR-PassManager-Crash-Reproducer>: ") << error;
  return module.emitError()
         << "A failure has been detected while processing the MLIR module, a "
            "reproducer has been generated in '"
         << *crashReproducerFileName << "'";
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::PassManager(MLIRContext *ctx, Nesting nesting)
    : OpPassManager(Identifier::get(ModuleOp::getOperationName(), ctx),
                    nesting),
      context(ctx), passTiming(false), localReproducer(false),
      verifyPasses(true) {}

PassManager::~PassManager() {}

void PassManager::enableVerifier(bool enabled) { verifyPasses = enabled; }

/// Run the passes within this manager on the provided module.
LogicalResult PassManager::run(ModuleOp module) {
  // Before running, make sure to coalesce any adjacent pass adaptors in the
  // pipeline.
  getImpl().coalesceAdjacentAdaptorPasses();

  // Register all dialects for the current pipeline.
  DialectRegistry dependentDialects;
  getDependentDialects(dependentDialects);
  dependentDialects.loadAll(module.getContext());

  // Construct an analysis manager for the pipeline.
  ModuleAnalysisManager am(module, instrumentor.get());

  // Notify the context that we start running a pipeline for book keeping.
  module.getContext()->enterMultiThreadedExecution();

  // If reproducer generation is enabled, run the pass manager with crash
  // handling enabled.
  LogicalResult result = crashReproducerFileName
                             ? runWithCrashRecovery(module, am)
                             : OpToOpPassAdaptor::runPipeline(
                                   getPasses(), module, am, verifyPasses);

  // Notify the context that the run is done.
  module.getContext()->exitMultiThreadedExecution();

  // Dump all of the pass statistics if necessary.
  if (passStatisticsMode)
    dumpStatistics();
  return result;
}

/// Enable support for the pass manager to generate a reproducer on the event
/// of a crash or a pass failure. `outputFile` is a .mlir filename used to write
/// the generated reproducer. If `genLocalReproducer` is true, the pass manager
/// will attempt to generate a local reproducer that contains the smallest
/// pipeline.
void PassManager::enableCrashReproducerGeneration(StringRef outputFile,
                                                  bool genLocalReproducer) {
  crashReproducerFileName = std::string(outputFile);
  localReproducer = genLocalReproducer;
}

/// Add the provided instrumentation to the pass manager.
void PassManager::addInstrumentation(std::unique_ptr<PassInstrumentation> pi) {
  if (!instrumentor)
    instrumentor = std::make_unique<PassInstrumentor>();

  instrumentor->addInstrumentation(std::move(pi));
}

//===----------------------------------------------------------------------===//
// AnalysisManager
//===----------------------------------------------------------------------===//

/// Returns a pass instrumentation object for the current operation.
PassInstrumentor *AnalysisManager::getPassInstrumentor() const {
  ParentPointerT curParent = parent;
  while (auto *parentAM = curParent.dyn_cast<const AnalysisManager *>())
    curParent = parentAM->parent;
  return curParent.get<const ModuleAnalysisManager *>()->getPassInstrumentor();
}

/// Get an analysis manager for the given child operation.
AnalysisManager AnalysisManager::nest(Operation *op) {
  auto it = impl->childAnalyses.find(op);
  if (it == impl->childAnalyses.end())
    it = impl->childAnalyses
             .try_emplace(op, std::make_unique<NestedAnalysisMap>(op))
             .first;
  return {this, it->second.get()};
}

/// Invalidate any non preserved analyses.
void detail::NestedAnalysisMap::invalidate(
    const detail::PreservedAnalyses &pa) {
  // If all analyses were preserved, then there is nothing to do here.
  if (pa.isAll())
    return;

  // Invalidate the analyses for the current operation directly.
  analyses.invalidate(pa);

  // If no analyses were preserved, then just simply clear out the child
  // analysis results.
  if (pa.isNone()) {
    childAnalyses.clear();
    return;
  }

  // Otherwise, invalidate each child analysis map.
  SmallVector<NestedAnalysisMap *, 8> mapsToInvalidate(1, this);
  while (!mapsToInvalidate.empty()) {
    auto *map = mapsToInvalidate.pop_back_val();
    for (auto &analysisPair : map->childAnalyses) {
      analysisPair.second->invalidate(pa);
      if (!analysisPair.second->childAnalyses.empty())
        mapsToInvalidate.push_back(analysisPair.second.get());
    }
  }
}

//===----------------------------------------------------------------------===//
// PassInstrumentation
//===----------------------------------------------------------------------===//

PassInstrumentation::~PassInstrumentation() {}

//===----------------------------------------------------------------------===//
// PassInstrumentor
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct PassInstrumentorImpl {
  /// Mutex to keep instrumentation access thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// Set of registered instrumentations.
  std::vector<std::unique_ptr<PassInstrumentation>> instrumentations;
};
} // end namespace detail
} // end namespace mlir

PassInstrumentor::PassInstrumentor() : impl(new PassInstrumentorImpl()) {}
PassInstrumentor::~PassInstrumentor() {}

/// See PassInstrumentation::runBeforePipeline for details.
void PassInstrumentor::runBeforePipeline(
    Identifier name,
    const PassInstrumentation::PipelineParentInfo &parentInfo) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePipeline(name, parentInfo);
}

/// See PassInstrumentation::runAfterPipeline for details.
void PassInstrumentor::runAfterPipeline(
    Identifier name,
    const PassInstrumentation::PipelineParentInfo &parentInfo) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPipeline(name, parentInfo);
}

/// See PassInstrumentation::runBeforePass for details.
void PassInstrumentor::runBeforePass(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePass(pass, op);
}

/// See PassInstrumentation::runAfterPass for details.
void PassInstrumentor::runAfterPass(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPass(pass, op);
}

/// See PassInstrumentation::runAfterPassFailed for details.
void PassInstrumentor::runAfterPassFailed(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPassFailed(pass, op);
}

/// See PassInstrumentation::runBeforeAnalysis for details.
void PassInstrumentor::runBeforeAnalysis(StringRef name, TypeID id,
                                         Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforeAnalysis(name, id, op);
}

/// See PassInstrumentation::runAfterAnalysis for details.
void PassInstrumentor::runAfterAnalysis(StringRef name, TypeID id,
                                        Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterAnalysis(name, id, op);
}

/// Add the given instrumentation to the collection.
void PassInstrumentor::addInstrumentation(
    std::unique_ptr<PassInstrumentation> pi) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  impl->instrumentations.emplace_back(std::move(pi));
}
