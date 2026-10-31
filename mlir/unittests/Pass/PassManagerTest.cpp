//===- PassManagerTest.cpp - PassManager unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

using namespace mlir;
using namespace mlir::detail;

namespace {
/// Analysis that operates on any operation.
struct GenericAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericAnalysis)

  GenericAnalysis(Operation *op) : isFunc(isa<func::FuncOp>(op)) {}
  const bool isFunc;
};

/// Analysis that operates on a specific operation.
struct OpSpecificAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSpecificAnalysis)

  OpSpecificAnalysis(func::FuncOp op) : isSecret(op.getName() == "secret") {}
  const bool isSecret;
};

/// Simple pass to annotate a func::FuncOp with the results of analysis.
struct AnnotateFunctionPass
    : public PassWrapper<AnnotateFunctionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateFunctionPass)

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    Builder builder(op->getParentOfType<ModuleOp>());

    auto &ga = getAnalysis<GenericAnalysis>();
    auto &sa = getAnalysis<OpSpecificAnalysis>();

    op->setDiscardableAttr("isFunc", builder.getBoolAttr(ga.isFunc));
    op->setDiscardableAttr("isSecret", builder.getBoolAttr(sa.isSecret));
  }
};

TEST(PassManagerTest, OpSpecificAnalysis) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create a module with 2 functions.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (StringRef name : {"secret", "not_secret"}) {
    auto func = func::FuncOp::create(builder.getUnknownLoc(), name,
                                     builder.getFunctionType({}, {}));
    func.setPrivate();
    module->push_back(func);
  }

  // Instantiate and run our pass.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addNestedPass<func::FuncOp>(std::make_unique<AnnotateFunctionPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Verify that each function got annotated with expected attributes.
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    ASSERT_TRUE(isa<BoolAttr>(func->getDiscardableAttr("isFunc")));
    EXPECT_TRUE(cast<BoolAttr>(func->getDiscardableAttr("isFunc")).getValue());

    bool isSecret = func.getName() == "secret";
    ASSERT_TRUE(isa<BoolAttr>(func->getDiscardableAttr("isSecret")));
    EXPECT_EQ(cast<BoolAttr>(func->getDiscardableAttr("isSecret")).getValue(),
              isSecret);
  }
}

/// Simple pass to annotate a func::FuncOp with a single attribute `didProcess`.
struct AddAttrFunctionPass
    : public PassWrapper<AddAttrFunctionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddAttrFunctionPass)

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    Builder builder(op->getParentOfType<ModuleOp>());
    if (op->hasDiscardableAttr("didProcess"))
      op->setDiscardableAttr("didProcessAgain", builder.getUnitAttr());

    // We always want to set this one.
    op->setDiscardableAttr("didProcess", builder.getUnitAttr());
  }
};

/// Simple pass to annotate a func::FuncOp with a single attribute
/// `didProcess2`.
struct AddSecondAttrFunctionPass
    : public PassWrapper<AddSecondAttrFunctionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddSecondAttrFunctionPass)

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    Builder builder(op->getParentOfType<ModuleOp>());
    op->setDiscardableAttr("didProcess2", builder.getUnitAttr());
  }
};

/// PassInstrumentation to count pass callbacks and signal pass failures.
struct TestPassInstrumentation : public PassInstrumentation {
  int beforePassCallbackCount = 0;
  int afterPassCallbackCount = 0;
  int afterPassFailedCallbackCount = 0;

  bool failBeforePass = false;
  bool failAfterPass = false;

  void runBeforePass(Pass *pass, Operation *op) override {
    if (pass->getTypeID() != TypeID::get<AddAttrFunctionPass>())
      return;

    ++beforePassCallbackCount;
    if (failBeforePass)
      signalPassFailure(pass);
  }
  void runAfterPass(Pass *pass, Operation *op) override {
    if (pass->getTypeID() != TypeID::get<AddAttrFunctionPass>())
      return;

    ++afterPassCallbackCount;
    if (failAfterPass)
      signalPassFailure(pass);
  }
  void runAfterPassFailed(Pass *pass, Operation *op) override {
    if (pass->getTypeID() != TypeID::get<AddAttrFunctionPass>())
      return;

    ++afterPassFailedCallbackCount;
  }
};

TEST(PassManagerTest, PassInstrumentation) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder b(&context);

  // Create a module with 1 function.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  auto func = func::FuncOp::create(b.getUnknownLoc(), "test_func",
                                   b.getFunctionType({}, {}));
  func.setPrivate();
  module->push_back(func);

  struct InstrumentationCounts {
    int beforePass;
    int afterPass;
    int afterPassFailed;
  };

  auto runInstrumentation =
      [&](bool failBefore,
          bool failAfter) -> std::pair<LogicalResult, InstrumentationCounts> {
    // Instantiate and run our pass.
    auto pm = PassManager::on<ModuleOp>(&context);
    auto instrumentation = std::make_unique<TestPassInstrumentation>();
    auto *instrumentationPtr = instrumentation.get();
    instrumentation->failBeforePass = failBefore;
    instrumentation->failAfterPass = failAfter;
    pm.addInstrumentation(std::move(instrumentation));
    pm.addNestedPass<func::FuncOp>(std::make_unique<AddAttrFunctionPass>());
    LogicalResult result = pm.run(module.get());

    InstrumentationCounts counts = {
        instrumentationPtr->beforePassCallbackCount,
        instrumentationPtr->afterPassCallbackCount,
        instrumentationPtr->afterPassFailedCallbackCount};
    return {result, counts};
  };

  for (bool failBefore : {false, true}) {
    for (bool failAfter : {false, true}) {
      auto [result, counts] = runInstrumentation(failBefore, failAfter);

      InstrumentationCounts expected;
      if (failBefore) {
        EXPECT_TRUE(failed(result))
            << "failBefore=" << failBefore << ", failAfter=" << failAfter;
        expected = {/*beforePass=*/1, /*afterPass=*/0, /*afterPassFailed=*/1};
      } else if (failAfter) {
        EXPECT_TRUE(failed(result))
            << "failBefore=" << failBefore << ", failAfter=" << failAfter;
        expected = {/*beforePass=*/1, /*afterPass=*/1, /*afterPassFailed=*/0};
      } else {
        EXPECT_TRUE(succeeded(result))
            << "failBefore=" << failBefore << ", failAfter=" << failAfter;
        expected = {/*beforePass=*/1, /*afterPass=*/1, /*afterPassFailed=*/0};
      }

      EXPECT_EQ(counts.beforePass, expected.beforePass)
          << "failBefore=" << failBefore << ", failAfter=" << failAfter;
      EXPECT_EQ(counts.afterPass, expected.afterPass)
          << "failBefore=" << failBefore << ", failAfter=" << failAfter;
      EXPECT_EQ(counts.afterPassFailed, expected.afterPassFailed)
          << "failBefore=" << failBefore << ", failAfter=" << failAfter;
    }
  }
}

// A worker keeps exclusive ownership of its state even while nested waits let
// the same thread execute other tasks from the outer loop.
TEST(PassManagerTest, ParallelWorkerOwnership) {
  llvm::StdThreadPool pool(llvm::hardware_concurrency(4));
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.setThreadPool(pool);
  std::vector<std::atomic<bool>> active(pool.getMaxConcurrency());
  for (auto &flag : active)
    flag = false;
  std::atomic<unsigned> visits = 0;
  auto indices = llvm::seq<unsigned>(0, 64);
  EXPECT_TRUE(succeeded(failableParallelForEachWithWorkerId(
      &context, indices.begin(), indices.end(), [&](unsigned, unsigned worker) {
        EXPECT_LT(worker, active.size());
        EXPECT_FALSE(active[worker].exchange(true));
        auto inner = llvm::seq<unsigned>(0, 8);
        EXPECT_TRUE(
            succeeded(failableParallelForEach(&context, inner, [&](unsigned) {
              ++visits;
              return success();
            })));
        EXPECT_TRUE(active[worker].exchange(false));
        return success();
      })));
  EXPECT_EQ(visits.load(), 64u * 8);
  EXPECT_TRUE(failed(failableParallelForEachWithWorkerId(
      &context, indices.begin(), indices.end(),
      [&](unsigned, unsigned) { return failure(); })));
  EXPECT_TRUE(succeeded(failableParallelForEachWithWorkerId(
      &context, indices.begin(), indices.begin(), [&](unsigned, unsigned) {
        ADD_FAILURE() << "empty range must not invoke the callback";
        return failure();
      })));

  context.disableMultithreading();
  unsigned serialVisits = 0;
  EXPECT_TRUE(succeeded(failableParallelForEachWithWorkerId(
      &context, indices.begin(), indices.end(), [&](unsigned, unsigned worker) {
        EXPECT_EQ(worker, 0u);
        ++serialVisits;
        return success();
      })));
  EXPECT_EQ(serialVisits, 64u);
  EXPECT_TRUE(failed(failableParallelForEachWithWorkerId(
      &context, indices.begin(), indices.end(),
      [&](unsigned, unsigned) { return failure(); })));
}

struct ConcurrentInstrumentation : PassInstrumentation {
  bool isThreadSafe() const override { return true; }
  std::mutex mutex;
  std::condition_variable cv;
  unsigned entered = 0;
  unsigned overlapped = 0;

  void runBeforePass(Pass *, Operation *) override {
    std::unique_lock<std::mutex> lock(mutex);
    ++entered;
    cv.notify_all();
    overlapped += cv.wait_for(lock, std::chrono::seconds(10),
                              [&] { return entered == 2; });
  }
};

TEST(PassManagerTest, ConcurrentInstrumentation) {
  if (!llvm::llvm_is_multithreaded())
    GTEST_SKIP();
  PassInstrumentor instrumentor;
  auto instrumentation = std::make_unique<ConcurrentInstrumentation>();
  auto *ptr = instrumentation.get();
  instrumentor.addInstrumentation(std::move(instrumentation));
  std::thread first([&] { instrumentor.runBeforePass(nullptr, nullptr); });
  std::thread second([&] { instrumentor.runBeforePass(nullptr, nullptr); });
  first.join();
  second.join();
  EXPECT_EQ(ptr->entered, 2u);
  EXPECT_EQ(ptr->overlapped, 2u);
}

// Non-atomic callback state remains protected, including when another
// instrumentation opts into concurrent dispatch.
TEST(PassManagerTest, SerializedInstrumentation) {
  if (!llvm::llvm_is_multithreaded())
    GTEST_SKIP();
  struct Counter : PassInstrumentation {
    Counter(bool threadSafe) : threadSafe(threadSafe) {}
    bool threadSafe;
    unsigned calls = 0;
    bool isThreadSafe() const override { return threadSafe; }
    void runBeforePass(Pass *, Operation *) override {
      // Without serialization, this non-atomic read-modify-write would race.
      // Yield between the read and write to increase the chance of a lost
      // update, making the final count too small. ThreadSanitizer can also
      // detect the race even if the final count happens to be correct.
      unsigned previous = calls;
      std::this_thread::yield();
      calls = previous + 1;
    }
  };
  PassInstrumentor instrumentor;
  auto counter = std::make_unique<Counter>(false);
  auto concurrent = std::make_unique<Counter>(true);
  auto *counterPtr = counter.get();
  auto *concurrentPtr = concurrent.get();
  instrumentor.addInstrumentation(std::move(counter));
  instrumentor.addInstrumentation(std::move(concurrent));
  std::array<std::thread, 4> threads;
  for (auto &thread : threads)
    thread = std::thread([&] {
      for (unsigned i = 0; i < 128; ++i)
        instrumentor.runBeforePass(nullptr, nullptr);
    });
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(counterPtr->calls, 512u);
  EXPECT_EQ(concurrentPtr->calls, 512u);
}

TEST(PassManagerTest, CustomTimingManagerSerialization) {
  struct CustomTimingManager : TimingManager {
    std::atomic<bool> inCallback = false;
    std::atomic<bool> overlap = false;

    void observeCallback() {
      if (inCallback.exchange(true))
        overlap = true;
      std::this_thread::yield();
      inCallback = false;
    }
    std::optional<void *> rootTimer() override { return this; }
    void startTimer(void *) override { observeCallback(); }
    void stopTimer(void *) override { observeCallback(); }
    void *nestTimer(void *, const void *,
                    function_ref<std::string()>) override {
      observeCallback();
      return this;
    }
  };

  llvm::StdThreadPool pool(llvm::hardware_concurrency(4));
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.setThreadPool(pool);
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);
  OwningOpRef<ModuleOp> module(ModuleOp::create(builder.getUnknownLoc()));
  for (unsigned i = 0; i < 64; ++i) {
    auto func =
        func::FuncOp::create(builder.getUnknownLoc(), "f" + std::to_string(i),
                             builder.getFunctionType({}, {}));
    func.setPrivate();
    module->push_back(func);
  }
  auto timing = std::make_unique<CustomTimingManager>();
  auto *timingPtr = timing.get();
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addNestedPass<func::FuncOp>(std::make_unique<AnnotateFunctionPass>());
  pm.enableTiming(std::move(timing));
  EXPECT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_FALSE(timingPtr->overlap.load());
}

TEST(PassManagerTest, NestedParallelTiming) {
  llvm::StdThreadPool pool(llvm::hardware_concurrency(4));
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.setThreadPool(pool);
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  OwningOpRef<ModuleOp> owner(module);
  for (unsigned i = 0; i < 8; ++i) {
    auto nested = ModuleOp::create(builder.getUnknownLoc());
    module.push_back(nested);
    for (unsigned j = 0; j < 8; ++j) {
      auto func =
          func::FuncOp::create(builder.getUnknownLoc(), "f" + std::to_string(j),
                               builder.getFunctionType({}, {}));
      func.setPrivate();
      nested.push_back(func);
    }
  }
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.nest<ModuleOp>().addNestedPass<func::FuncOp>(
      std::make_unique<AnnotateFunctionPass>());
  pm.enableTiming();
  for (unsigned run = 0; run < 2; ++run) {
    ASSERT_TRUE(succeeded(pm.run(module)));
    unsigned processed = 0;
    module.walk([&](func::FuncOp func) {
      EXPECT_TRUE(func->hasDiscardableAttr("isFunc"));
      ++processed;
    });
    EXPECT_EQ(processed, 64u);
  }
}

TEST(PassManagerTest, ExecutionAction) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create a module with 2 functions.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  auto f = func::FuncOp::create(builder.getUnknownLoc(), "process_me_once",
                                builder.getFunctionType({}, {}));
  f.setPrivate();
  module->push_back(f);

  // Instantiate our passes.
  auto pm = PassManager::on<ModuleOp>(&context);
  auto pass = std::make_unique<AddAttrFunctionPass>();
  auto *passPtr = pass.get();
  pm.addNestedPass<func::FuncOp>(std::move(pass));
  pm.addNestedPass<func::FuncOp>(std::make_unique<AddSecondAttrFunctionPass>());
  // Duplicate the first pass to ensure that we *only* run the *first* pass, not
  // all instances of this pass kind. Notice that this pass (and the test as a
  // whole) are built to ensure that we can run just a single pass out of a
  // pipeline that may contain duplicates.
  pm.addNestedPass<func::FuncOp>(std::make_unique<AddAttrFunctionPass>());

  // Use the action manager to only hit the first pass, not the second one.
  auto onBreakpoint = [&](const tracing::ActionActiveStack *backtrace)
      -> tracing::ExecutionContext::Control {
    // Not a PassExecutionAction, apply the action.
    auto *passExec = dyn_cast<PassExecutionAction>(&backtrace->getAction());
    if (!passExec)
      return tracing::ExecutionContext::Next;

    // If this isn't a function, apply the action.
    if (!isa<func::FuncOp>(passExec->getOp()))
      return tracing::ExecutionContext::Next;

    // Only apply the first function pass. Not all instances of the first pass,
    // only the first pass.
    if (passExec->getPass().getThreadingSiblingOrThis() == passPtr)
      return tracing::ExecutionContext::Next;

    // Do not apply any other passes in the pass manager.
    return tracing::ExecutionContext::Skip;
  };

  // Set up our breakpoint manager.
  tracing::TagBreakpointManager simpleManager;
  tracing::ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(PassExecutionAction::tag);

  // Register the execution context in the MLIRContext.
  context.registerActionHandler(executionCtx);

  // Run the pass manager, expecting our handler to be called.
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Verify that each function got annotated with `didProcess` and *not*
  // `didProcess2`.
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    ASSERT_TRUE(func->getDiscardableAttr("didProcess"));
    ASSERT_FALSE(func->getDiscardableAttr("didProcess2"));
    ASSERT_FALSE(func->getDiscardableAttr("didProcessAgain"));
  }
}

namespace {
struct InvalidPass : Pass {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InvalidPass)

  InvalidPass() : Pass(TypeID::get<InvalidPass>(), StringRef("invalid_op")) {}
  StringRef getName() const override { return "Invalid Pass"; }
  void runOnOperation() override {}
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<InvalidPass>(
        *static_cast<const InvalidPass *>(this));
  }
};
} // namespace

TEST(PassManagerTest, InvalidPass) {
  MLIRContext context;
  context.allowUnregisteredDialects();

  // Create a module
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

  // Add a single "invalid_op" operation
  OpBuilder builder(&module->getBodyRegion());
  OperationState state(UnknownLoc::get(&context), "invalid_op");
  builder.insert(Operation::create(state));

  // Register a diagnostic handler to capture the diagnostic so that we can
  // check it later.
  std::unique_ptr<Diagnostic> diagnostic;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    diagnostic = std::make_unique<Diagnostic>(std::move(diag));
  });

  // Instantiate and run our pass.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.nest("invalid_op").addPass(std::make_unique<InvalidPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(failed(result));
  ASSERT_TRUE(diagnostic.get() != nullptr);
  EXPECT_EQ(
      diagnostic->str(),
      "'invalid_op' op trying to schedule a pass on an unregistered operation");

  // Check that clearing the pass manager effectively removed the pass.
  pm.clear();
  result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Check that adding the pass at the top-level triggers a fatal error.
  ASSERT_DEATH(pm.addPass(std::make_unique<InvalidPass>()),
               "Can't add pass 'Invalid Pass' restricted to 'invalid_op' on a "
               "PassManager intended to run on 'builtin.module', did you "
               "intend to nest?");
}

/// Simple pass to annotate a func::FuncOp with the results of analysis.
struct InitializeCheckingPass
    : public PassWrapper<InitializeCheckingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitializeCheckingPass)
  LogicalResult initialize(MLIRContext *ctx) final {
    initialized = true;
    return success();
  }
  bool initialized = false;

  void runOnOperation() override {
    if (!initialized) {
      getOperation()->emitError() << "Pass isn't initialized!";
      signalPassFailure();
    }
  }
};

TEST(PassManagerTest, PassInitialization) {
  MLIRContext context;
  context.allowUnregisteredDialects();

  // Create a module
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

  // Instantiate and run our pass.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addPass(std::make_unique<InitializeCheckingPass>());
  EXPECT_TRUE(succeeded(pm.run(module.get())));

  // Adding a second copy of the pass, we should also initialize it!
  pm.addPass(std::make_unique<InitializeCheckingPass>());
  EXPECT_TRUE(succeeded(pm.run(module.get())));
}

} // namespace
