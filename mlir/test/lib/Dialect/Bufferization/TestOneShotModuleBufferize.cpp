//===- TestOneShotModuleBufferzation.cpp - Bufferization Test -----*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "TestAttributes.h" // TestTensorEncodingAttr, TestMemRefLayoutAttr
#include "TestDialect.h"

using namespace mlir;

namespace {
MemRefLayoutAttrInterface
getMemRefLayoutForTensorEncoding(RankedTensorType tensorType) {
  if (auto encoding = dyn_cast_if_present<test::TestTensorEncodingAttr>(
          tensorType.getEncoding())) {
    return cast<MemRefLayoutAttrInterface>(test::TestMemRefLayoutAttr::get(
        tensorType.getContext(), encoding.getDummy()));
  }
  return {};
}

struct TestOneShotModuleBufferizePass
    : public PassWrapper<TestOneShotModuleBufferizePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOneShotModuleBufferizePass)

  TestOneShotModuleBufferizePass() = default;
  TestOneShotModuleBufferizePass(const TestOneShotModuleBufferizePass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const final {
    return "test-one-shot-module-bufferize";
  }
  StringRef getDescription() const final {
    return "Pass to test One Shot Module Bufferization";
  }

  void runOnOperation() override {

    llvm::errs() << "Running TestOneShotModuleBufferize on: "
                 << getOperation()->getName() << "\n";
    bufferization::OneShotBufferizationOptions opt;

    opt.bufferizeFunctionBoundaries = true;
    opt.functionArgTypeConverterFn =
        [&](bufferization::TensorLikeType tensor, Attribute memSpace,
            func::FuncOp, const bufferization::BufferizationOptions &) {
          assert(isa<RankedTensorType>(tensor) && "tests only builtin tensors");
          auto tensorType = cast<RankedTensorType>(tensor);
          auto layout = getMemRefLayoutForTensorEncoding(tensorType);
          return cast<bufferization::BufferLikeType>(
              MemRefType::get(tensorType.getShape(),
                              tensorType.getElementType(), layout, memSpace));
        };

    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::runOneShotModuleBufferize(getOperation(), opt,
                                                        bufferizationState)))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::test {
void registerTestOneShotModuleBufferizePass() {
  PassRegistration<TestOneShotModuleBufferizePass>();
}
} // namespace mlir::test
