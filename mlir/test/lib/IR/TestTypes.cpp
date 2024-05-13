//===- TestTypes.cpp - Test passes for MLIR types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTypes.h"
#include "TestDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace test;

namespace {
struct TestRecursiveTypesPass
    : public PassWrapper<TestRecursiveTypesPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRecursiveTypesPass)

  LogicalResult createIRWithTypes();

  StringRef getArgument() const final { return "test-recursive-types"; }
  StringRef getDescription() const final {
    return "Test support for recursive types";
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Just make sure recursive types are printed and parsed.
    if (func.getName() == "roundtrip")
      return;

    // Create a recursive type and print it as a part of a dummy op.
    if (func.getName() == "create") {
      if (failed(createIRWithTypes()))
        signalPassFailure();
      return;
    }

    // Unknown key.
    func.emitOpError() << "unexpected function name";
    signalPassFailure();
  }
};
} // namespace

LogicalResult TestRecursiveTypesPass::createIRWithTypes() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();
  auto type = TestRecursiveType::get(ctx, "some_long_and_unique_name");
  if (failed(type.setBody(type)))
    return func.emitError("expected to be able to set the type body");

  // Setting the same body is fine.
  if (failed(type.setBody(type)))
    return func.emitError(
        "expected to be able to set the type body to the same value");

  // Setting a different body is not.
  if (succeeded(type.setBody(IndexType::get(ctx))))
    return func.emitError(
        "not expected to be able to change function body more than once");

  // Expecting to get the same type for the same name.
  auto other = TestRecursiveType::get(ctx, "some_long_and_unique_name");
  if (type != other)
    return func.emitError("expected type name to be the uniquing key");

  // Create the op to check how the type is printed.
  OperationState state(func.getLoc(), "test.dummy_type_test_op");
  state.addTypes(type);
  func.getBody().front().push_front(Operation::create(state));

  return success();
}

namespace mlir {
namespace test {

void registerTestRecursiveTypesPass() {
  PassRegistration<TestRecursiveTypesPass>();
}

} // namespace test
} // namespace mlir
