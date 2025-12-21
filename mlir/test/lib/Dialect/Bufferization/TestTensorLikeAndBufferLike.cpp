//===- TestTensorLikeAndBufferLike.cpp - Bufferization Test -----*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include <string>

using namespace mlir;

namespace {
std::string getImplementationStatus(Type type) {
  if (isa<bufferization::TensorLikeType>(type)) {
    return "is_tensor_like";
  }
  if (isa<bufferization::BufferLikeType>(type)) {
    return "is_buffer_like";
  }
  return {};
}

DictionaryAttr findAllImplementeesOfTensorOrBufferLike(func::FuncOp funcOp) {
  llvm::SmallVector<NamedAttribute> attributes;

  const auto funcType = funcOp.getFunctionType();
  for (auto [index, inputType] : llvm::enumerate(funcType.getInputs())) {
    const auto status = getImplementationStatus(inputType);
    if (status.empty()) {
      continue;
    }

    attributes.push_back(
        NamedAttribute(StringAttr::get(funcOp.getContext(),
                                       "operand_" + std::to_string(index)),
                       StringAttr::get(funcOp.getContext(), status)));
  }

  for (auto [index, resultType] : llvm::enumerate(funcType.getResults())) {
    const auto status = getImplementationStatus(resultType);
    if (status.empty()) {
      continue;
    }

    attributes.push_back(NamedAttribute(
        StringAttr::get(funcOp.getContext(), "result_" + std::to_string(index)),
        StringAttr::get(funcOp.getContext(), status)));
  }

  return mlir::DictionaryAttr::get(funcOp.getContext(), attributes);
}

/// This pass tests whether specified types implement TensorLike and (or)
/// BufferLike type interfaces defined in bufferization.
///
/// The pass analyses operation signature. When the aforementioned interface
/// implementation found, an attribute is added to the operation, signifying the
/// associated operand / result.
struct TestTensorLikeAndBufferLikePass
    : public PassWrapper<TestTensorLikeAndBufferLikePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTensorLikeAndBufferLikePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, test::TestDialect>();
  }
  StringRef getArgument() const final { return "test-tensorlike-bufferlike"; }
  StringRef getDescription() const final {
    return "Module pass to test custom types that implement TensorLike / "
           "BufferLike interfaces";
  }

  void runOnOperation() override {
    auto op = getOperation();

    op.walk([](func::FuncOp funcOp) {
      const auto dict = findAllImplementeesOfTensorOrBufferLike(funcOp);
      if (!dict.empty()) {
        funcOp->setAttr("found", dict);
      }
    });
  }
};
} // namespace

namespace mlir::test {
void registerTestTensorLikeAndBufferLikePass() {
  PassRegistration<TestTensorLikeAndBufferLikePass>();
}
} // namespace mlir::test
