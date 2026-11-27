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

#include "TestAttributesCore.h" // TestTensorEncodingAttr, TestMemRefLayoutAttr
#include "TestOpsInterfaces.h"
#include "TestTypesCore.h" // TestTensorType, TestMemrefType

#include "llvm/ADT/TypeSwitch.h"

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

test::TestMemRefLayoutAttr
getLayoutFromBuffer(bufferization::BufferLikeType buffer) {
  auto layout =
      llvm::TypeSwitch<bufferization::BufferLikeType,
                       MemRefLayoutAttrInterface>(buffer)
          .Case([&](MemRefType memref) { return memref.getLayout(); })
          .Case([&](test::TestMemrefType testMemref) {
            return cast<MemRefLayoutAttrInterface>(testMemref.getLayout());
          })
          .Default([](bufferization::BufferLikeType) {
            return MemRefLayoutAttrInterface();
          });
  return dyn_cast_or_null<test::TestMemRefLayoutAttr>(layout);
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

  Option<bool> allowReturnAllocsFromLoops{
      *this, "allow-return-allocs-from-loops",
      llvm::cl::desc("Allows returning/yielding new allocations from a loop."),
      llvm::cl::init(false)};
  Option<bool> testTensorToMemRef{
      *this, "test-tensor-to-memref",
      llvm::cl::desc("Convert test tensors to builtin memrefs."),
      llvm::cl::init(false)};
  Option<bool> testTensorToDynamicMemRef{
      *this, "test-tensor-to-dynamic-memref",
      llvm::cl::desc("Convert test tensors to builtin memrefs with dynamic "
                     "extents."),
      llvm::cl::init(false)};

  void runOnOperation() override {

    llvm::errs() << "Running TestOneShotModuleBufferize on: "
                 << getOperation()->getName() << "\n";
    bufferization::OneShotBufferizationOptions opt;

    opt.bufferizeFunctionBoundaries = true;
    opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
    opt.functionArgTypeConverterFn =
        [&](bufferization::TensorLikeType tensor, Attribute memSpace,
            func::FuncOp, const bufferization::BufferizationOptions &options) {
          return options.unknownTypeConverterFn(tensor, memSpace, options);
        };
    opt.unknownTypeConverterFn =
        [&](bufferization::TensorLikeType tensor, Attribute memSpace,
            const bufferization::BufferizationOptions &options) {
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
                if (testTensorToDynamicMemRef)
                  return cast<bufferization::BufferLikeType>(MemRefType::get(
                      SmallVector<int64_t>(testTensorType.getRank(),
                                           ShapedType::kDynamic),
                      testTensorType.getElementType(),
                      MemRefLayoutAttrInterface{}, memSpace));
                if (testTensorToMemRef)
                  return cast<bufferization::BufferLikeType>(
                      MemRefType::get(testTensorType.getShape(),
                                      testTensorType.getElementType(),
                                      MemRefLayoutAttrInterface{}, memSpace));
                assert(memSpace == nullptr &&
                       "memory space is not supported for test types");
                // Note: unknown type conversion cannot create layout because
                // test tensor does not have an encoding, this is probably
                // enough for a test.
                return test::TestMemrefType::get(
                    testTensorType.getContext(), testTensorType.getShape(),
                    testTensorType.getElementType(), /*layout=*/nullptr);
              })
              .Case([&](test::TestTensorLikeWrapperType wrapper) {
                return options.unknownTypeConverterFn(
                    cast<bufferization::TensorLikeType>(wrapper.getInner()),
                    memSpace, options);
              })
              .Default([&](bufferization::TensorLikeType tensor) {
                llvm_unreachable("unexpected tensor type");
                return bufferization::BufferLikeType{};
              });
        };
    // A simple yet distinct (from upstream) policy: compare layouts and return
    // "smaller" one.
    opt.reconcileBufferTypeMismatchFn =
        [](bufferization::BufferLikeType x, bufferization::BufferLikeType y,
           const bufferization::BufferizationOptions &)
        -> FailureOr<bufferization::BufferLikeType> {
      auto lhsLayout = getLayoutFromBuffer(x);
      auto rhsLayout = getLayoutFromBuffer(y);
      if (lhsLayout && rhsLayout) {
        return lhsLayout.getDummy().getValue() <=
                       rhsLayout.getDummy().getValue()
                   ? x
                   : y;
      }
      return rhsLayout ? y : x;
    };
    opt.castFn = [&](OpBuilder &b, Location loc, Type dest, Value value) {
      return test::TestDummyCastOp::create(b, loc, dest, value).getResult();
    };
    // Function signature update only works with memref.cast. Disable it to
    // align behaviour for upstream and user casts.
    opt.inferFunctionResultLayout = false;

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
