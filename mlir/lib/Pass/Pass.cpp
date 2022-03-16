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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Mutex.h"
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
  OpPassManagerImpl(OperationName opName, OpPassManager::Nesting nesting)
      : name(opName.getStringRef()), opName(opName),
        initializationGeneration(0), nesting(nesting) {}
  OpPassManagerImpl(StringRef name, OpPassManager::Nesting nesting)
      : name(name), initializationGeneration(0), nesting(nesting) {}

  /// Merge the passes of this pass manager into the one provided.
  void mergeInto(OpPassManagerImpl &rhs);

  /// Nest a new operation pass manager for the given operation kind under this
  /// pass manager.
  OpPassManager &nest(StringAttr nestedName);
  OpPassManager &nest(StringRef nestedName);

  /// Add the given pass to this pass manager. If this pass has a concrete
  /// operation type, it must be the same type as this pass manager.
  void addPass(std::unique_ptr<Pass> pass);

  /// Clear the list of passes in this pass manager, other options are
  /// preserved.
  void clear();

  /// Finalize the pass list in preparation for execution. This includes
  /// coalescing adjacent pass managers when possible, verifying scheduled
  /// passes, etc.
  LogicalResult finalizePassList(MLIRContext *ctx);

  /// Return the operation name of this pass manager.
  OperationName getOpName(MLIRContext &context) {
    if (!opName)
      opName = OperationName(name, &context);
    return *opName;
  }

  /// The name of the operation that passes of this pass manager operate on.
  std::string name;

  /// The cached OperationName (internalized in the context) for the name of the
  /// operation that passes of this pass manager operate on.
  Optional<OperationName> opName;

  /// The set of passes to run as part of this pass manager.
  std::vector<std::unique_ptr<Pass>> passes;

  /// The current initialization generation of this pass manager. This is used
  /// to indicate when a pass manager should be reinitialized.
  unsigned initializationGeneration;

  /// Control the implicit nesting of passes that mismatch the name set for this
  /// OpPassManager.
  OpPassManager::Nesting nesting;
};
} // namespace detail
} // namespace mlir

void OpPassManagerImpl::mergeInto(OpPassManagerImpl &rhs) {
  assert(name == rhs.name && "merging unrelated pass managers");
  for (auto &pass : passes)
    rhs.passes.push_back(std::move(pass));
  passes.clear();
}

OpPassManager &OpPassManagerImpl::nest(StringAttr nestedName) {
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

void OpPassManagerImpl::clear() { passes.clear(); }

LogicalResult OpPassManagerImpl::finalizePassList(MLIRContext *ctx) {
  // Walk the pass list and merge adjacent adaptors.
  OpToOpPassAdaptor *lastAdaptor = nullptr;
  for (auto &pass : passes) {
    // Check to see if this pass is an adaptor.
    if (auto *currentAdaptor = dyn_cast<OpToOpPassAdaptor>(pass.get())) {
      // If it is the first adaptor in a possible chain, remember it and
      // continue.
      if (!lastAdaptor) {
        lastAdaptor = currentAdaptor;
        continue;
      }

      // Otherwise, merge into the existing adaptor and delete the current one.
      currentAdaptor->mergeInto(*lastAdaptor);
      pass.reset();
    } else if (lastAdaptor) {
      // If this pass is not an adaptor, then finalize and forget any existing
      // adaptor.
      for (auto &pm : lastAdaptor->getPassManagers())
        if (failed(pm.getImpl().finalizePassList(ctx)))
          return failure();
      lastAdaptor = nullptr;
    }
  }

  // If there was an adaptor at the end of the manager, finalize it as well.
  if (lastAdaptor) {
    for (auto &pm : lastAdaptor->getPassManagers())
      if (failed(pm.getImpl().finalizePassList(ctx)))
        return failure();
  }

  // Now that the adaptors have been merged, erase any empty slots corresponding
  // to the merged adaptors that were nulled-out in the loop above.
  Optional<RegisteredOperationName> opName =
      getOpName(*ctx).getRegisteredInfo();
  llvm::erase_if(passes, std::logical_not<std::unique_ptr<Pass>>());

  // Verify that all of the passes are valid for the operation.
  for (std::unique_ptr<Pass> &pass : passes) {
    if (opName && !pass->canScheduleOn(*opName)) {
      return emitError(UnknownLoc::get(ctx))
             << "unable to schedule pass '" << pass->getName()
             << "' on a PassManager intended to run on '" << name << "'!";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// OpPassManager
//===----------------------------------------------------------------------===//

OpPassManager::OpPassManager(StringAttr name, Nesting nesting)
    : impl(new OpPassManagerImpl(name, nesting)) {}
OpPassManager::OpPassManager(StringRef name, Nesting nesting)
    : impl(new OpPassManagerImpl(name, nesting)) {}
OpPassManager::OpPassManager(OpPassManager &&rhs) : impl(std::move(rhs.impl)) {}
OpPassManager::OpPassManager(const OpPassManager &rhs) { *this = rhs; }
OpPassManager &OpPassManager::operator=(const OpPassManager &rhs) {
  impl = std::make_unique<OpPassManagerImpl>(rhs.impl->name, rhs.impl->nesting);
  impl->initializationGeneration = rhs.impl->initializationGeneration;
  for (auto &pass : rhs.impl->passes) {
    auto newPass = pass->clone();
    newPass->threadingSibling = pass.get();
    impl->passes.push_back(std::move(newPass));
  }
  return *this;
}

OpPassManager::~OpPassManager() = default;

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
OpPassManager &OpPassManager::nest(StringAttr nestedName) {
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

void OpPassManager::clear() { impl->clear(); }

/// Returns the number of passes held by this manager.
size_t OpPassManager::size() const { return impl->passes.size(); }

/// Returns the internal implementation instance.
OpPassManagerImpl &OpPassManager::getImpl() { return *impl; }

/// Return the operation name that this pass manager operates on.
StringRef OpPassManager::getOpName() const { return impl->name; }

/// Return the operation name that this pass manager operates on.
OperationName OpPassManager::getOpName(MLIRContext &context) const {
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

void OpPassManager::setNesting(Nesting nesting) { impl->nesting = nesting; }

OpPassManager::Nesting OpPassManager::getNesting() { return impl->nesting; }

LogicalResult OpPassManager::initialize(MLIRContext *context,
                                        unsigned newInitGeneration) {
  if (impl->initializationGeneration == newInitGeneration)
    return success();
  impl->initializationGeneration = newInitGeneration;
  for (Pass &pass : getPasses()) {
    // If this pass isn't an adaptor, directly initialize it.
    auto *adaptor = dyn_cast<OpToOpPassAdaptor>(&pass);
    if (!adaptor) {
      if (failed(pass.initialize(context)))
        return failure();
      continue;
    }

    // Otherwise, initialize each of the adaptors pass managers.
    for (OpPassManager &adaptorPM : adaptor->getPassManagers())
      if (failed(adaptorPM.initialize(context, newInitGeneration)))
        return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

LogicalResult OpToOpPassAdaptor::run(Pass *pass, Operation *op,
                                     AnalysisManager am, bool verifyPasses,
                                     unsigned parentInitGeneration) {
  if (!op->isRegistered())
    return op->emitOpError()
           << "trying to schedule a pass on an unregistered operation";
  if (!op->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return op->emitOpError() << "trying to schedule a pass on an operation not "
                                "marked as 'IsolatedFromAbove'";

  // Initialize the pass state with a callback for the pass to dynamically
  // execute a pipeline on the currently visited operation.
  PassInstrumentor *pi = am.getPassInstrumentor();
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        pass};
  auto dynamicPipelineCallback = [&](OpPassManager &pipeline,
                                     Operation *root) -> LogicalResult {
    if (!op->isAncestor(root))
      return root->emitOpError()
             << "Trying to schedule a dynamic pipeline on an "
                "operation that isn't "
                "nested under the current operation the pass is processing";
    assert(pipeline.getOpName() == root->getName().getStringRef());

    // Before running, finalize the passes held by the pipeline.
    if (failed(pipeline.getImpl().finalizePassList(root->getContext())))
      return failure();

    // Initialize the user provided pipeline and execute the pipeline.
    if (failed(pipeline.initialize(root->getContext(), parentInitGeneration)))
      return failure();
    AnalysisManager nestedAm = root == op ? am : am.nest(root);
    return OpToOpPassAdaptor::runPipeline(pipeline.getPasses(), root, nestedAm,
                                          verifyPasses, parentInitGeneration,
                                          pi, &parentInfo);
  };
  pass->passState.emplace(op, am, dynamicPipelineCallback);

  // Instrument before the pass has run.
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

  // When verifyPasses is specified, we run the verifier (unless the pass
  // failed).
  if (!passFailed && verifyPasses) {
    bool runVerifierNow = true;

    // If the pass is an adaptor pass, we don't run the verifier recursively
    // because the nested operations should have already been verified after
    // nested passes had run.
    bool runVerifierRecursively = !isa<OpToOpPassAdaptor>(pass);

    // Reduce compile time by avoiding running the verifier if the pass didn't
    // change the IR since the last time the verifier was run:
    //
    //  1) If the pass said that it preserved all analyses then it can't have
    //     permuted the IR.
    //
    // We run these checks in EXPENSIVE_CHECKS mode out of caution.
#ifndef EXPENSIVE_CHECKS
    runVerifierNow = !pass->passState->preservedAnalyses.isAll();
#endif
    if (runVerifierNow)
      passFailed = failed(verify(op, runVerifierRecursively));
  }

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
    AnalysisManager am, bool verifyPasses, unsigned parentInitGeneration,
    PassInstrumentor *instrumentor,
    const PassInstrumentation::PipelineParentInfo *parentInfo) {
  assert((!instrumentor || parentInfo) &&
         "expected parent info if instrumentor is provided");
  auto scopeExit = llvm::make_scope_exit([&] {
    // Clear out any computed operation analyses. These analyses won't be used
    // any more in this pipeline, and this helps reduce the current working set
    // of memory. If preserving these analyses becomes important in the future
    // we can re-evaluate this.
    am.clear();
  });

  // Run the pipeline over the provided operation.
  if (instrumentor)
    instrumentor->runBeforePipeline(op->getName().getIdentifier(), *parentInfo);
  for (Pass &pass : passes)
    if (failed(run(&pass, op, am, verifyPasses, parentInitGeneration)))
      return failure();
  if (instrumentor)
    instrumentor->runAfterPipeline(op->getName().getIdentifier(), *parentInfo);
  return success();
}

/// Find an operation pass manager that can operate on an operation of the given
/// type, or nullptr if one does not exist.
static OpPassManager *findPassManagerFor(MutableArrayRef<OpPassManager> mgrs,
                                         StringRef name) {
  auto *it = llvm::find_if(
      mgrs, [&](OpPassManager &mgr) { return mgr.getOpName() == name; });
  return it == mgrs.end() ? nullptr : &*it;
}

/// Find an operation pass manager that can operate on an operation of the given
/// type, or nullptr if one does not exist.
static OpPassManager *findPassManagerFor(MutableArrayRef<OpPassManager> mgrs,
                                         OperationName name,
                                         MLIRContext &context) {
  auto *it = llvm::find_if(
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
        auto *mgr = findPassManagerFor(mgrs, op.getName(), *op.getContext());
        if (!mgr)
          continue;

        // Run the held pipeline over the current operation.
        unsigned initGeneration = mgr->impl->initializationGeneration;
        if (failed(runPipeline(mgr->getPasses(), &op, am.nest(&op),
                               verifyPasses, initGeneration, instrumentor,
                               &parentInfo)))
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
  MLIRContext *context = &getContext();

  // Create the async executors if they haven't been created, or if the main
  // pipeline has changed.
  if (asyncExecutors.empty() || hasSizeMismatch(asyncExecutors.front(), mgrs))
    asyncExecutors.assign(context->getThreadPool().getThreadCount(), mgrs);

  // Run a prepass over the operation to collect the nested operations to
  // execute over. This ensures that an analysis manager exists for each
  // operation, as well as providing a queue of operations to execute over.
  std::vector<std::pair<Operation *, AnalysisManager>> opAMPairs;
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        // Add this operation iff the name matches any of the pass managers.
        if (findPassManagerFor(mgrs, op.getName(), *context))
          opAMPairs.emplace_back(&op, am.nest(&op));
      }
    }
  }

  // Get the current thread for this adaptor.
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        this};
  auto *instrumentor = am.getPassInstrumentor();

  // An atomic failure variable for the async executors.
  std::vector<std::atomic<bool>> activePMs(asyncExecutors.size());
  std::fill(activePMs.begin(), activePMs.end(), false);
  auto processFn = [&](auto &opPMPair) {
    // Find a pass manager for this operation.
    auto it = llvm::find_if(activePMs, [](std::atomic<bool> &isActive) {
      bool expectedInactive = false;
      return isActive.compare_exchange_strong(expectedInactive, true);
    });
    unsigned pmIndex = it - activePMs.begin();

    // Get the pass manager for this operation and execute it.
    auto *pm = findPassManagerFor(asyncExecutors[pmIndex],
                                  opPMPair.first->getName(), *context);
    assert(pm && "expected valid pass manager for operation");

    unsigned initGeneration = pm->impl->initializationGeneration;
    LogicalResult pipelineResult =
        runPipeline(pm->getPasses(), opPMPair.first, opPMPair.second,
                    verifyPasses, initGeneration, instrumentor, &parentInfo);

    // Reset the active bit for this pass manager.
    activePMs[pmIndex].store(false);
    return pipelineResult;
  };

  // Signal a failure if any of the executors failed.
  if (failed(failableParallelForEach(context, opAMPairs, processFn)))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::PassManager(MLIRContext *ctx, Nesting nesting,
                         StringRef operationName)
    : OpPassManager(StringAttr::get(ctx, operationName), nesting), context(ctx),
      initializationKey(DenseMapInfo<llvm::hash_code>::getTombstoneKey()),
      passTiming(false), verifyPasses(true) {}

PassManager::~PassManager() = default;

void PassManager::enableVerifier(bool enabled) { verifyPasses = enabled; }

/// Run the passes within this manager on the provided operation.
LogicalResult PassManager::run(Operation *op) {
  MLIRContext *context = getContext();
  assert(op->getName() == getOpName(*context) &&
         "operation has a different name than the PassManager or is from a "
         "different context");

  // Register all dialects for the current pipeline.
  DialectRegistry dependentDialects;
  getDependentDialects(dependentDialects);
  context->appendDialectRegistry(dependentDialects);
  for (StringRef name : dependentDialects.getDialectNames())
    context->getOrLoadDialect(name);

  // Before running, make sure to finalize the pipeline pass list.
  if (failed(getImpl().finalizePassList(context)))
    return failure();

  // Initialize all of the passes within the pass manager with a new generation.
  llvm::hash_code newInitKey = context->getRegistryHash();
  if (newInitKey != initializationKey) {
    if (failed(initialize(context, impl->initializationGeneration + 1)))
      return failure();
    initializationKey = newInitKey;
  }

  // Construct a top level analysis manager for the pipeline.
  ModuleAnalysisManager am(op, instrumentor.get());

  // Notify the context that we start running a pipeline for book keeping.
  context->enterMultiThreadedExecution();

  // If reproducer generation is enabled, run the pass manager with crash
  // handling enabled.
  LogicalResult result =
      crashReproGenerator ? runWithCrashRecovery(op, am) : runPasses(op, am);

  // Notify the context that the run is done.
  context->exitMultiThreadedExecution();

  // Dump all of the pass statistics if necessary.
  if (passStatisticsMode)
    dumpStatistics();
  return result;
}

/// Add the provided instrumentation to the pass manager.
void PassManager::addInstrumentation(std::unique_ptr<PassInstrumentation> pi) {
  if (!instrumentor)
    instrumentor = std::make_unique<PassInstrumentor>();

  instrumentor->addInstrumentation(std::move(pi));
}

LogicalResult PassManager::runPasses(Operation *op, AnalysisManager am) {
  return OpToOpPassAdaptor::runPipeline(getPasses(), op, am, verifyPasses,
                                        impl->initializationGeneration);
}

//===----------------------------------------------------------------------===//
// AnalysisManager
//===----------------------------------------------------------------------===//

/// Get an analysis manager for the given operation, which must be a proper
/// descendant of the current operation represented by this analysis manager.
AnalysisManager AnalysisManager::nest(Operation *op) {
  Operation *currentOp = impl->getOperation();
  assert(currentOp->isProperAncestor(op) &&
         "expected valid descendant operation");

  // Check for the base case where the provided operation is immediately nested.
  if (currentOp == op->getParentOp())
    return nestImmediate(op);

  // Otherwise, we need to collect all ancestors up to the current operation.
  SmallVector<Operation *, 4> opAncestors;
  do {
    opAncestors.push_back(op);
    op = op->getParentOp();
  } while (op != currentOp);

  AnalysisManager result = *this;
  for (Operation *op : llvm::reverse(opAncestors))
    result = result.nestImmediate(op);
  return result;
}

/// Get an analysis manager for the given immediately nested child operation.
AnalysisManager AnalysisManager::nestImmediate(Operation *op) {
  assert(impl->getOperation() == op->getParentOp() &&
         "expected immediate child operation");

  auto it = impl->childAnalyses.find(op);
  if (it == impl->childAnalyses.end())
    it = impl->childAnalyses
             .try_emplace(op, std::make_unique<NestedAnalysisMap>(op, impl))
             .first;
  return {it->second.get()};
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

PassInstrumentation::~PassInstrumentation() = default;

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
} // namespace detail
} // namespace mlir

PassInstrumentor::PassInstrumentor() : impl(new PassInstrumentorImpl()) {}
PassInstrumentor::~PassInstrumentor() = default;

/// See PassInstrumentation::runBeforePipeline for details.
void PassInstrumentor::runBeforePipeline(
    StringAttr name,
    const PassInstrumentation::PipelineParentInfo &parentInfo) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePipeline(name, parentInfo);
}

/// See PassInstrumentation::runAfterPipeline for details.
void PassInstrumentor::runAfterPipeline(
    StringAttr name,
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
