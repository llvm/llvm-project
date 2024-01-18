//===- TestParametricSpecialization.cpp - Pass for metaprog
// specialization--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ParametricSpecializationOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/ParametricSpecialization.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <mutex>
#include <utility>

#define DEBUG_TYPE "parametric-specialization"

using namespace mlir;

namespace {

struct SpecializingRequest {
  /// Op to specialize
  ParametricOpInterface targetOp;
  /// The arguments to specialize it with.
  DictionaryAttr metaArgs;
  /// The "callers" to update
  SmallVector<SpecializingOpInterface, 0> callers;
  /// The operation post-specialization
  OwningOpRef<ParametricOpInterface> specialized;
};

struct TestParametricSpecializationPass
    : public PassWrapper<TestParametricSpecializationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestParametricSpecializationPass)

  StringRef getArgument() const final {
    return "test-parametric-specialization";
  }
  StringRef getDescription() const final {
    return "Test the parametric specialization of parametric programs.";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (!op->hasTrait<OpTrait::SymbolTable>()) {
      op->emitOpError()
          << getArgument()
          << " pass can only run on an operation that defines a SymbolTable";
      signalPassFailure();
    }
    OpBuilder builder(op->getContext());
    SymbolTable symTab(op);

    MLIRContext &ctx = getContext();

    // Walk the body of the module, and find "roots": operations that are
    // already specialized. We'll use these as "roots" to specialize the
    // parametric ones.
    SmallVector<Operation *> rootOps;
    for (Operation &nestedOp : op->getRegion(0).getOps()) {
      if (!isa<ParametricOpInterface>(nestedOp))
        rootOps.push_back(&nestedOp);
    }

    std::map<StringRef, SpecializingRequest> specializationRequests;
    std::mutex tasksMutex;
    LogicalResult result = success();

    // Run in parallel on every root, and for each, walk the body and find
    // "calls" to functions that need specialization.
    result = failableParallelForEach(&ctx, rootOps, [&](Operation *root) {
      auto result = root->walk([&](Operation *innerOp) {
        auto specializingOp = dyn_cast<SpecializingOpInterface>(innerOp);
        if (!specializingOp)
          return WalkResult::advance();
        auto targetNameAttr = specializingOp.getTarget();
        auto targetOp = symTab.lookup(targetNameAttr.getRootReference());
        if (!targetOp) {
          innerOp->emitOpError()
              << "can't find target '" << targetNameAttr << "' in SymbolTable";
          return WalkResult::interrupt();
        }
        auto parametricTargetOp = dyn_cast<ParametricOpInterface>(targetOp);
        if (!parametricTargetOp) {
          auto diag = targetOp->emitOpError();
          diag << "expected target to implement 'ParametricOpInterface'";
          diag.attachNote() << "while specializing " << *innerOp;
          return WalkResult::interrupt();
        }
        auto metaArgs = specializingOp.getMetaArgs();
        auto failureOrMangledName = parametricTargetOp.getMangledName(metaArgs);
        if (failed(failureOrMangledName)) {
          parametricTargetOp->emitOpError()
              << "failed to mangled with meta args " << metaArgs;
          return WalkResult::interrupt();
        }
        StringAttr mangledName = *failureOrMangledName;
        std::unique_lock<std::mutex> lock(tasksMutex);
        auto &request = specializationRequests[mangledName.getValue()];
        if (request.targetOp && request.targetOp != targetOp) {
          auto diag = targetOp->emitOpError();
          diag << "unexpected mangling collision while specializing with "
                  "meta args "
               << metaArgs << ", mangled name " << mangledName;
          diag.attachNote() << "while specializing " << *innerOp;
          return WalkResult::interrupt();
        }
        request.targetOp = parametricTargetOp;
        request.metaArgs = metaArgs;
        request.callers.push_back(specializingOp);
        LLVM_DEBUG({ llvm::errs() << "Request for " << mangledName << "\n"; });
        return WalkResult::advance();
      });
      return success(!result.wasInterrupted());
    });
    if (failed(result)) {
      signalPassFailure();
      return;
    }
    LLVM_DEBUG({
      llvm::errs() << "Got " << specializationRequests.size() << " requests\n";
    });

    std::map<StringRef,
             std::pair<SmallVector<Operation *, 0> *, OwningOpRef<Operation *>>>
        specializationResults;
    result = failableParallelForEach(
        &ctx, specializationRequests,
        [&](std::pair<const llvm::StringRef, SpecializingRequest> &request) {
          ParametricOpInterface targetOp = request.second.targetOp;
          DictionaryAttr metaArgs = request.second.metaArgs;
          OwningOpRef<ParametricOpInterface> specializedOp(targetOp.clone());
          if (failed(specializedOp->specialize(metaArgs))) {
            std::unique_lock<std::mutex> lock(tasksMutex);
            targetOp->emitOpError() << "failed to specialize with " << metaArgs;
            return failure();
          }
          request.second.specialized = std::move(specializedOp);
          return success();
        });
    if (failed(result)) {
      signalPassFailure();
      return;
    }

    llvm::ThreadPool &threadPool = ctx.getThreadPool();
    llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
    for (auto &request : specializationRequests) {
      Operation *op = request.second.specialized.get();
      symTab.insert(op);
      LLVM_DEBUG({
        llvm::errs() << "Inserted " << cast<SymbolOpInterface>(op).getName()
                     << "\n";
      });
      tasksGroup.async([&] {
        Operation *op = request.second.specialized.release();
        auto specializedOp = cast<SymbolOpInterface>(op);
        for (SpecializingOpInterface caller : request.second.callers) {
          if (failed(caller.setSpecializedTarget(specializedOp))) {
            std::unique_lock<std::mutex> lock(tasksMutex);
            caller->emitOpError() << "failed to specialize\n";
            signalPassFailure();
          }
        }
      });
    }
    tasksGroup.wait();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestParametricSpecializationPass() {
  PassRegistration<TestParametricSpecializationPass>();
}
} // namespace test
} // namespace mlir
