//===- TestPointerLikeTypeInterface.cpp - Test PointerLikeType interface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for testing the OpenACC PointerLikeType
// interface methods.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::acc;

namespace {

struct OperationTracker : public OpBuilder::Listener {
  SmallVector<Operation *> insertedOps;

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    insertedOps.push_back(op);
  }
};

struct TestPointerLikeTypeInterfacePass
    : public PassWrapper<TestPointerLikeTypeInterfacePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPointerLikeTypeInterfacePass)

  TestPointerLikeTypeInterfacePass() = default;
  TestPointerLikeTypeInterfacePass(const TestPointerLikeTypeInterfacePass &pass)
      : PassWrapper(pass) {
    testMode = pass.testMode;
  }

  Pass::Option<std::string> testMode{
      *this, "test-mode",
      llvm::cl::desc("Test mode: walk, alloc, copy, free, load, or store"),
      llvm::cl::init("walk")};

  StringRef getArgument() const override {
    return "test-acc-pointer-like-interface";
  }

  StringRef getDescription() const override {
    return "Test OpenACC PointerLikeType interface methods on any implementing "
           "type";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<acc::OpenACCDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

private:
  void walkAndPrint();
  void testGenAllocate(Operation *op, Value result, PointerLikeType pointerType,
                       OpBuilder &builder);
  void testGenFree(Operation *op, Value result, PointerLikeType pointerType,
                   OpBuilder &builder);
  void testGenCopy(Operation *srcOp, Operation *destOp, Value srcResult,
                   Value destResult, PointerLikeType pointerType,
                   OpBuilder &builder);
  void testGenLoad(Operation *op, Value result, PointerLikeType pointerType,
                   OpBuilder &builder);
  void testGenStore(Operation *op, Value result, PointerLikeType pointerType,
                    OpBuilder &builder, Value providedValue = {});

  struct PointerCandidate {
    Operation *op;
    Value result;
    PointerLikeType pointerType;
  };
};

void TestPointerLikeTypeInterfacePass::runOnOperation() {
  if (testMode == "walk") {
    walkAndPrint();
    return;
  }

  auto func = getOperation();
  OpBuilder builder(&getContext());

  if (testMode == "alloc" || testMode == "free" || testMode == "load" ||
      testMode == "store") {
    // Collect all candidates first
    SmallVector<PointerCandidate> candidates;
    // For store mode, also look for a test value to use
    Value testValue;
    func.walk([&](Operation *op) {
      if (op->hasAttr("test.ptr")) {
        for (auto result : op->getResults()) {
          if (isa<PointerLikeType>(result.getType())) {
            candidates.push_back(
                {op, result, cast<PointerLikeType>(result.getType())});
            break; // Only take the first PointerLikeType result
          }
        }
      }
      // Collect value marked with test.value for store tests
      if (testMode == "store" && op->hasAttr("test.value")) {
        if (op->getNumResults() > 0)
          testValue = op->getResult(0);
      }
    });

    // Now test all candidates
    for (const auto &candidate : candidates) {
      if (testMode == "alloc")
        testGenAllocate(candidate.op, candidate.result, candidate.pointerType,
                        builder);
      else if (testMode == "free")
        testGenFree(candidate.op, candidate.result, candidate.pointerType,
                    builder);
      else if (testMode == "load")
        testGenLoad(candidate.op, candidate.result, candidate.pointerType,
                    builder);
      else if (testMode == "store")
        testGenStore(candidate.op, candidate.result, candidate.pointerType,
                     builder, testValue);
    }
  } else if (testMode == "copy") {
    // Collect all source and destination candidates
    SmallVector<PointerCandidate> sources, destinations;

    func.walk([&](Operation *op) {
      if (op->hasAttr("test.src_ptr")) {
        for (auto result : op->getResults()) {
          if (isa<PointerLikeType>(result.getType())) {
            sources.push_back(
                {op, result, cast<PointerLikeType>(result.getType())});
            break;
          }
        }
      }
      if (op->hasAttr("test.dest_ptr")) {
        for (auto result : op->getResults()) {
          if (isa<PointerLikeType>(result.getType())) {
            destinations.push_back(
                {op, result, cast<PointerLikeType>(result.getType())});
            break;
          }
        }
      }
    });

    // Try copying from each source to each destination
    for (const auto &src : sources)
      for (const auto &dest : destinations)
        testGenCopy(src.op, dest.op, src.result, dest.result, src.pointerType,
                    builder);
  }
}

void TestPointerLikeTypeInterfacePass::walkAndPrint() {
  auto func = getOperation();

  func.walk([&](Operation *op) {
    // Look for operations marked with "test.ptr", "test.src_ptr", or
    // "test.dest_ptr"
    if (op->hasAttr("test.ptr") || op->hasAttr("test.src_ptr") ||
        op->hasAttr("test.dest_ptr")) {
      llvm::errs() << "Operation: ";
      op->print(llvm::errs());
      llvm::errs() << "\n";

      // Check each result to see if it's a PointerLikeType
      for (auto result : op->getResults()) {
        if (isa<PointerLikeType>(result.getType())) {
          llvm::errs() << "  Result " << result.getResultNumber()
                       << " is PointerLikeType: ";
          result.getType().print(llvm::errs());
          llvm::errs() << "\n";
        } else {
          llvm::errs() << "  Result " << result.getResultNumber()
                       << " is NOT PointerLikeType: ";
          result.getType().print(llvm::errs());
          llvm::errs() << "\n";
        }
      }

      if (op->getNumResults() == 0)
        llvm::errs() << "  Operation has no results\n";

      llvm::errs() << "\n";
    }
  });
}

void TestPointerLikeTypeInterfacePass::testGenAllocate(
    Operation *op, Value result, PointerLikeType pointerType,
    OpBuilder &builder) {
  Location loc = op->getLoc();

  // Create a new builder with the listener and set insertion point
  OperationTracker tracker;
  OpBuilder newBuilder(op->getContext());
  newBuilder.setListener(&tracker);
  newBuilder.setInsertionPointAfter(op);

  // Call the genAllocate API
  bool needsFree = false;
  Value allocRes = pointerType.genAllocate(newBuilder, loc, "test_alloc",
                                           result.getType(), result, needsFree);

  if (allocRes) {
    llvm::errs() << "Successfully generated alloc for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "\tneeds free: " << (needsFree ? "true" : "false") << "\n";

    // Print all operations that were inserted
    for (Operation *insertedOp : tracker.insertedOps) {
      llvm::errs() << "\tGenerated: ";
      insertedOp->print(llvm::errs());
      llvm::errs() << "\n";
    }
  } else {
    llvm::errs() << "Failed to generate alloc for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
  }
}

void TestPointerLikeTypeInterfacePass::testGenFree(Operation *op, Value result,
                                                   PointerLikeType pointerType,
                                                   OpBuilder &builder) {
  Location loc = op->getLoc();

  // Create a new builder with the listener and set insertion point
  OperationTracker tracker;
  OpBuilder newBuilder(op->getContext());
  newBuilder.setListener(&tracker);
  newBuilder.setInsertionPointAfter(op);

  // Call the genFree API
  auto typedResult = cast<TypedValue<PointerLikeType>>(result);
  bool success = pointerType.genFree(newBuilder, loc, typedResult, result,
                                     result.getType());

  if (success) {
    llvm::errs() << "Successfully generated free for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";

    // Print all operations that were inserted
    for (Operation *insertedOp : tracker.insertedOps) {
      llvm::errs() << "\tGenerated: ";
      insertedOp->print(llvm::errs());
      llvm::errs() << "\n";
    }
  } else {
    llvm::errs() << "Failed to generate free for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
  }
}

void TestPointerLikeTypeInterfacePass::testGenCopy(
    Operation *srcOp, Operation *destOp, Value srcResult, Value destResult,
    PointerLikeType pointerType, OpBuilder &builder) {
  Location loc = destOp->getLoc();

  // Create a new builder with the listener and set insertion point
  OperationTracker tracker;
  OpBuilder newBuilder(destOp->getContext());
  newBuilder.setListener(&tracker);
  newBuilder.setInsertionPointAfter(destOp);

  // Call the genCopy API with the provided source and destination
  auto typedSrc = cast<TypedValue<PointerLikeType>>(srcResult);
  auto typedDest = cast<TypedValue<PointerLikeType>>(destResult);
  bool success = pointerType.genCopy(newBuilder, loc, typedDest, typedSrc,
                                     srcResult.getType());

  if (success) {
    llvm::errs() << "Successfully generated copy from source: ";
    srcOp->print(llvm::errs());
    llvm::errs() << " to destination: ";
    destOp->print(llvm::errs());
    llvm::errs() << "\n";

    // Print all operations that were inserted
    for (Operation *insertedOp : tracker.insertedOps) {
      llvm::errs() << "\tGenerated: ";
      insertedOp->print(llvm::errs());
      llvm::errs() << "\n";
    }
  } else {
    llvm::errs() << "Failed to generate copy from source: ";
    srcOp->print(llvm::errs());
    llvm::errs() << " to destination: ";
    destOp->print(llvm::errs());
    llvm::errs() << "\n";
  }
}

void TestPointerLikeTypeInterfacePass::testGenLoad(Operation *op, Value result,
                                                   PointerLikeType pointerType,
                                                   OpBuilder &builder) {
  Location loc = op->getLoc();

  // Create a new builder with the listener and set insertion point
  OperationTracker tracker;
  OpBuilder newBuilder(op->getContext());
  newBuilder.setListener(&tracker);
  newBuilder.setInsertionPointAfter(op);

  // Call the genLoad API
  auto typedResult = cast<TypedValue<PointerLikeType>>(result);
  Value loadRes = pointerType.genLoad(newBuilder, loc, typedResult, Type());

  if (loadRes) {
    llvm::errs() << "Successfully generated load for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "\tLoaded value type: ";
    loadRes.getType().print(llvm::errs());
    llvm::errs() << "\n";

    // Print all operations that were inserted
    for (Operation *insertedOp : tracker.insertedOps) {
      llvm::errs() << "\tGenerated: ";
      insertedOp->print(llvm::errs());
      llvm::errs() << "\n";
    }
  } else {
    llvm::errs() << "Failed to generate load for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
  }
}

void TestPointerLikeTypeInterfacePass::testGenStore(Operation *op, Value result,
                                                    PointerLikeType pointerType,
                                                    OpBuilder &builder,
                                                    Value providedValue) {
  Location loc = op->getLoc();

  // Create a new builder with the listener and set insertion point
  OperationTracker tracker;
  OpBuilder newBuilder(op->getContext());
  newBuilder.setListener(&tracker);
  newBuilder.setInsertionPointAfter(op);

  // Use provided value if available, otherwise create a constant
  Value valueToStore = providedValue;
  if (!valueToStore) {
    // Create a test value to store - use a constant matching the element type
    Type elementType = pointerType.getElementType();
    if (!elementType) {
      llvm::errs() << "Failed to generate store for operation: ";
      op->print(llvm::errs());
      llvm::errs() << "\n";
      return;
    }

    if (elementType.isIntOrIndex()) {
      auto attr = newBuilder.getIntegerAttr(elementType, 42);
      valueToStore =
          arith::ConstantOp::create(newBuilder, loc, elementType, attr);
    } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
      auto attr = newBuilder.getFloatAttr(floatType, 42.0);
      valueToStore =
          arith::ConstantOp::create(newBuilder, loc, floatType, attr);
    } else {
      llvm::errs() << "Failed to generate store for operation: ";
      op->print(llvm::errs());
      llvm::errs() << "\n";
      return;
    }
  }

  // Call the genStore API
  auto typedResult = cast<TypedValue<PointerLikeType>>(result);
  bool success =
      pointerType.genStore(newBuilder, loc, valueToStore, typedResult);

  if (success) {
    llvm::errs() << "Successfully generated store for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";

    // Print all operations that were inserted
    for (Operation *insertedOp : tracker.insertedOps) {
      llvm::errs() << "\tGenerated: ";
      insertedOp->print(llvm::errs());
      llvm::errs() << "\n";
    }
  } else {
    llvm::errs() << "Failed to generate store for operation: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {
void registerTestPointerLikeTypeInterfacePass() {
  PassRegistration<TestPointerLikeTypeInterfacePass>();
}
} // namespace test
} // namespace mlir
