//===- TestConcurrencyPasses.cpp - Passes for binding concurrency tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A test pass that meets other threads at a named native barrier.
// The Python bindings tests use it to prove that several
// PassManagers, on independent contexts, execute native pass code
// concurrently: the barrier can only be satisfied while all participating
// threads are inside Pass::runOnOperation at the same time, which requires
// PassManager.run to detach from the Python interpreter.
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace {
/// A single-use counting barrier.
class RendezvousBarrier {
public:
  /// Returns the process-global barrier named `id`, creating it on first use.
  static RendezvousBarrier &get(const std::string &id) {
    static std::mutex mapMutex;
    // std::map rather than an LLVM container: barriers must never move, and
    // the returned reference must stay valid across later insertions.
    static std::map<std::string, RendezvousBarrier> barriers;
    std::lock_guard<std::mutex> lock(mapMutex);
    return barriers[id];
  }

  /// Arrives at the barrier and blocks until `participants` threads have
  /// arrived. Returns std::nullopt on success. On timeout, or if the barrier
  /// was already poisoned, poisons the barrier and returns the number of
  /// arrivals observed.
  std::optional<unsigned> arriveAndWait(unsigned participants,
                                        std::chrono::seconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    if (!poisoned) {
      if (++arrived >= participants) {
        cv.notify_all();
        return std::nullopt;
      }
      bool satisfied = cv.wait_for(
          lock, timeout, [&] { return arrived >= participants || poisoned; });
      if (satisfied && !poisoned)
        return std::nullopt;
    }
    poisoned = true;
    cv.notify_all();
    return arrived;
  }

private:
  std::mutex mutex;
  std::condition_variable cv;

  /// Number of threads that have reached the barrier.
  unsigned arrived = 0;

  /// Set when a participant times out, so that latecomers fail fast instead
  /// of waiting out their own timeout.
  bool poisoned = false;
};

struct TestPythonRendezvousPass
    : public mlir::PassWrapper<TestPythonRendezvousPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPythonRendezvousPass)

  TestPythonRendezvousPass() = default;
  TestPythonRendezvousPass(const TestPythonRendezvousPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const final { return "test-python-rendezvous"; }
  llvm::StringRef getDescription() const final {
    return "Blocks natively until `participants` threads have entered this "
           "pass with the same barrier-id; fails after timeout-seconds.";
  }

  Option<std::string> barrierId{
      *this, "barrier-id",
      llvm::cl::desc("Name of the process-global barrier to meet at"),
      llvm::cl::init("default")};
  Option<unsigned> participants{
      *this, "participants",
      llvm::cl::desc("Number of threads that must arrive at the barrier"),
      llvm::cl::init(2)};
  Option<unsigned> timeoutSeconds{
      *this, "timeout-seconds",
      llvm::cl::desc("Seconds to wait for the other participants"),
      llvm::cl::init(20)};

  void runOnOperation() final {
    std::optional<unsigned> arrivedAtFailure =
        RendezvousBarrier::get(barrierId).arriveAndWait(
            participants, std::chrono::seconds(timeoutSeconds.getValue()));
    if (!arrivedAtFailure)
      return;
    getOperation().emitError()
        << "rendezvous '" << barrierId.getValue() << "' timed out with "
        << *arrivedAtFailure << " of " << participants.getValue()
        << " participants";
    signalPassFailure();
  }
};
} // namespace

extern "C" void mlirPythonTestRegisterRendezvousPass(void) {
  // Thread-safe and idempotent: registration happens once per process.
  static mlir::PassRegistration<TestPythonRendezvousPass> registration;
  (void)registration;
}
