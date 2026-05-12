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
            func::FuncOp, const bufferization::BufferizationOptions &options) {
          return options.unknownTypeConverterFn(tensor, memSpace, options);
        };
    opt.unknownTypeConverterFn =
        [&](bufferization::TensorLikeType tensor, Attribute memSpace,
            const bufferization::BufferizationOptions &) {
          return llvm::TypeSwitch<bufferization::TensorLikeType,
                                  bufferization::BufferLikeType>(tensor)
              .Case([&](UnrankedTensorType unrankedTensorType) {
                return cast<bufferization::BufferLikeType>(
                    UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                            memSpace));
              })
              .Case([&](RankedTensorType rankedTensorType) {
                // Note: builtin ranked tensor with custom encoding to layout
                // conversion.
                auto layout =
                    getMemRefLayoutForTensorEncoding(rankedTensorType);
                return cast<bufferization::BufferLikeType>(MemRefType::get(
                    rankedTensorType.getShape(),
                    rankedTensorType.getElementType(), layout, memSpace));
              })
              .Case([&](test::TestTensorType testTensorType)
                        -> bufferization::BufferLikeType {
                return test::TestMemrefType::get(
                    testTensorType.getContext(), testTensorType.getShape(),
                    testTensorType.getElementType(), memSpace);
              })
              .Default([&](bufferization::TensorLikeType tensor) {
                llvm_unreachable("unexpected tensor type");
                return bufferization::BufferLikeType{};
              });
        };
    // A simple yet distinct (from upstream) policy: compare layouts and return
    // "smaller" one.
    opt.reconcileBufferTypeMismatchFn =
        [](Operation *, bufferization::BufferLikeType x,
           bufferization::BufferLikeType y,
           const bufferization::BufferizationOptions &)
        -> FailureOr<bufferization::BufferLikeType> {
      auto getLayout = [](bufferization::BufferLikeType t) {
        auto m = dyn_cast<MemRefType>(t);
        return m ? dyn_cast<test::TestMemRefLayoutAttr>(m.getLayout())
                 : test::TestMemRefLayoutAttr();
      };
      auto lhsLayout = getLayout(x);
      auto rhsLayout = getLayout(y);
      if (lhsLayout && rhsLayout) {
        return lhsLayout.getDummy().getValue() <=
                       rhsLayout.getDummy().getValue()
                   ? x
                   : y;
      }
      return rhsLayout ? y : x;
    };
    opt.inferFunctionResultLayout = this->inferFunctionResultLayout;

    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::runOneShotModuleBufferize(getOperation(), opt,
                                                        bufferizationState)))
      signalPassFailure();
  }

  Option<bool> inferFunctionResultLayout{
      *this, "infer-function-result-layout",
      llvm::cl::desc(
          "Allows to change the function signature. By default, set to true."),
      llvm::cl::init(true)};
};
} // namespace

namespace mlir::test {
void registerTestOneShotModuleBufferizePass() {
  PassRegistration<TestOneShotModuleBufferizePass>();
}
} // namespace mlir::test
