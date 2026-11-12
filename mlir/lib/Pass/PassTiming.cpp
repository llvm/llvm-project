//===- PassTiming.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"

#include <deque>
#include <optional>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// PassTiming
//===----------------------------------------------------------------------===//

namespace {
struct PassTiming : public PassInstrumentation {
  PassTiming(TimingScope &timingScope) : rootScope(timingScope) {}
  PassTiming(std::unique_ptr<TimingManager> tm)
      : ownedTimingManager(std::move(tm)),
        ownedTimingScope(ownedTimingManager->getRootScope()),
        rootScope(ownedTimingScope) {}

  bool isThreadSafe() const override { return rootScope.isThreadSafe(); }

  /// Published adaptor scopes remain alive until their nested pipelines have
  /// joined. The deque below keeps their addresses stable when a worker helps
  /// execute another task while waiting for nested parallel work.
  DenseMap<PipelineParentInfo, TimingScope *> parentScopes;
  llvm::sys::SmartRWMutex<true> parentScopesMutex;

  /// The timing manager owned by this instrumentation (in case timing was
  /// enabled by the user on the pass manager without providing an external
  /// timing manager). This *must* appear before the `ownedTimingScope` to
  /// ensure the timing manager is destroyed *after* the scope, since the latter
  /// may hold a timer that points into the former.
  std::unique_ptr<TimingManager> ownedTimingManager;
  TimingScope ownedTimingScope;

  /// A stack of the currently active timing scopes per thread.
  ThreadLocalCache<std::deque<TimingScope>> activeThreadTimers;

  /// The root timing scope into which timing is reported.
  TimingScope &rootScope;

  //===--------------------------------------------------------------------===//
  // Pipeline
  //===--------------------------------------------------------------------===//

  void runBeforePipeline(std::optional<OperationName> name,
                         const PipelineParentInfo &parentInfo) override {
    auto &activeTimers = *activeThreadTimers;

    // Find the parent scope, either using the parent info or the root scope
    // (e.g. in the case of the top-level pipeline).
    TimingScope *parentScope = &rootScope;
    {
      llvm::sys::SmartScopedReader<true> lock(parentScopesMutex);
      auto it = parentScopes.find(parentInfo);
      if (it != parentScopes.end())
        parentScope = it->second;
    }

    // Use nullptr to anchor op-agnostic pipelines, otherwise use the name of
    // the operation.
    const void *timerId = name ? name->getAsOpaquePointer() : nullptr;
    activeTimers.push_back(parentScope->nest(timerId, [name] {
      return ("'" + (name ? name->getStringRef() : "any") + "' Pipeline").str();
    }));
  }

  void runAfterPipeline(std::optional<OperationName>,
                        const PipelineParentInfo &) override {
    auto &activeTimers = *activeThreadTimers;
    assert(!activeTimers.empty() && "expected active timer");
    activeTimers.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Pass
  //===--------------------------------------------------------------------===//

  void runBeforePass(Pass *pass, Operation *) override {
    auto &activeTimers = *activeThreadTimers;
    auto &parentScope = activeTimers.empty() ? rootScope : activeTimers.back();

    if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(pass)) {
      auto scope =
          parentScope.nest(pass->getThreadingSiblingOrThis(),
                           [adaptor]() { return adaptor->getAdaptorName(); });
      if (adaptor->getPassManagers().size() <= 1)
        scope.hide();
      activeTimers.push_back(std::move(scope));
      llvm::sys::SmartScopedWriter<true> lock(parentScopesMutex);
      parentScopes[{llvm::get_threadid(), pass}] = &activeTimers.back();
    } else {
      activeTimers.push_back(
          parentScope.nest(pass->getThreadingSiblingOrThis(),
                           [pass]() { return std::string(pass->getName()); }));
    }
  }

  void runAfterPass(Pass *pass, Operation *) override {
    if (isa<OpToOpPassAdaptor>(pass)) {
      llvm::sys::SmartScopedWriter<true> lock(parentScopesMutex);
      parentScopes.erase({llvm::get_threadid(), pass});
    }
    auto &activeTimers = *activeThreadTimers;
    assert(!activeTimers.empty() && "expected active timer");
    activeTimers.pop_back();
  }

  void runAfterPassFailed(Pass *pass, Operation *op) override {
    runAfterPass(pass, op);
  }

  //===--------------------------------------------------------------------===//
  // Analysis
  //===--------------------------------------------------------------------===//

  void runBeforeAnalysis(StringRef name, TypeID id, Operation *) override {
    auto &activeTimers = *activeThreadTimers;
    auto &parentScope = activeTimers.empty() ? rootScope : activeTimers.back();
    activeTimers.push_back(parentScope.nest(
        id.getAsOpaquePointer(), [name] { return "(A) " + name.str(); }));
  }

  void runAfterAnalysis(StringRef, TypeID, Operation *) override {
    auto &activeTimers = *activeThreadTimers;
    assert(!activeTimers.empty() && "expected active timer");
    activeTimers.pop_back();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

/// Add an instrumentation to time the execution of passes and the computation
/// of analyses.
void PassManager::enableTiming(TimingScope &timingScope) {
  if (!timingScope)
    return;
  addInstrumentation(std::make_unique<PassTiming>(timingScope));
}

/// Add an instrumentation to time the execution of passes and the computation
/// of analyses.
void PassManager::enableTiming(std::unique_ptr<TimingManager> tm) {
  if (!tm->getRootTimer())
    return; // no need to keep the timing manager around if it's disabled
  addInstrumentation(std::make_unique<PassTiming>(std::move(tm)));
}

/// Add an instrumentation to time the execution of passes and the computation
/// of analyses.
void PassManager::enableTiming() {
  auto tm = std::make_unique<DefaultTimingManager>();
  tm->setEnabled(true);
  enableTiming(std::move(tm));
}
