//===- InferConvolutionDimsTest.cpp - inferConvolutionDims unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

class InferConvolutionDimsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, func::FuncDialect>();
    ctx = std::make_unique<MLIRContext>(registry);
    ctx->loadAllAvailableDialects();
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> ctx;
};

/// Creates a Conv2DOp with loop order (d0, d1, d2, d3) where:
///   d0 = output height (oh), parallel
///   d1 = output width (ow), parallel
///   d2 = kernel height (kh), reduction
///   d3 = kernel width (kw), reduction
///
/// Indexing maps:
///   input:  (d0 + d2, d1 + d3)
///   filter: (d2, d3)
///   output: (d0, d1)
///
/// Semantic pairing: d0 <-> d2, d1 <-> d3
static linalg::Conv2DOp createConv2DOp(OpBuilder &builder, int64_t oh,
                                       int64_t ow, int64_t kh, int64_t kw) {
  Location loc = builder.getUnknownLoc();
  auto f32Type = builder.getF32Type();
  int64_t ih = oh + kh - 1;
  int64_t iw = ow + kw - 1;
  auto inputType = RankedTensorType::get({ih, iw}, f32Type);
  auto filterType = RankedTensorType::get({kh, kw}, f32Type);
  auto outputType = RankedTensorType::get({oh, ow}, f32Type);
  Value input = tensor::EmptyOp::create(builder, loc, inputType.getShape(),
                                        inputType.getElementType());
  Value filter = tensor::EmptyOp::create(builder, loc, filterType.getShape(),
                                         filterType.getElementType());
  Value output = tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                         outputType.getElementType());
  return linalg::Conv2DOp::create(
      builder, loc, outputType, ValueRange{input, filter}, ValueRange{output});
}

/// Creates a linalg.generic equivalent to the given Conv2DOp but with filter
/// loop dimensions swapped. The resulting op has loop order (d0, d1, d2, d3):
///   d0 = output height (oh), parallel
///   d1 = output width (ow), parallel
///   d2 = kernel width (kw), reduction  <-- swapped!
///   d3 = kernel height (kh), reduction <-- swapped!
///
/// Indexing maps:
///   input:  (d0 + d3, d1 + d2)
///   filter: (d2, d3)
///   output: (d0, d1)
///
/// Semantic pairing: d0 <-> d3, d1 <-> d2
static linalg::GenericOp
createConv2DWithSwappedFilterLoops(OpBuilder &builder,
                                   linalg::Conv2DOp conv2DOp) {
  Location loc = conv2DOp.getLoc();
  MLIRContext *ctx = builder.getContext();

  // Extract dimensions from the Conv2DOp. Require static shapes for simplicity.
  auto inputType = cast<RankedTensorType>(conv2DOp.getInputs()[0].getType());
  auto filterType = cast<RankedTensorType>(conv2DOp.getInputs()[1].getType());
  auto outputType = cast<RankedTensorType>(conv2DOp.getOutputs()[0].getType());
  assert(inputType.hasStaticShape() && "expected static input shape");
  assert(filterType.hasStaticShape() && "expected static filter shape");
  assert(outputType.hasStaticShape() && "expected static output shape");
  (void)outputType;
  int64_t kh = filterType.getDimSize(0);
  int64_t kw = filterType.getDimSize(1);

  // Filter dimensions are swapped: (kw, kh) instead of (kh, kw).
  auto f32Type = builder.getF32Type();
  auto swappedFilterType = RankedTensorType::get({kw, kh}, f32Type);
  Value input = tensor::EmptyOp::create(builder, loc, inputType.getShape(),
                                        inputType.getElementType());
  Value filter =
      tensor::EmptyOp::create(builder, loc, swappedFilterType.getShape(),
                              swappedFilterType.getElementType());
  Value output = tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                         outputType.getElementType());

  // Build indexing maps for swapped filter loop dimensions.
  // Original Conv2DOp: (d0=oh, d1=ow, d2=kh, d3=kw)
  // Swapped:           (d0=oh, d1=ow, d2=kw, d3=kh)
  AffineExpr d0, d1, d2, d3;
  bindDims(ctx, d0, d1, d2, d3);
  auto inputMap = AffineMap::get(4, 0, {d0 + d3, d1 + d2}, ctx);
  auto filterMap = AffineMap::get(4, 0, {d2, d3}, ctx);
  auto outputMap = AffineMap::get(4, 0, {d0, d1}, ctx);
  SmallVector<AffineMap> indexingMaps = {inputMap, filterMap, outputMap};
  SmallVector<utils::IteratorType> iterTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::reduction};
  return linalg::GenericOp::create(
      builder, loc, outputType, ValueRange{input, filter}, ValueRange{output},
      indexingMaps, iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
        Value add = arith::AddFOp::create(b, loc, args[2], mul);
        linalg::YieldOp::create(b, loc, add);
      });
}

TEST_F(InferConvolutionDimsTest, Conv2DPairing) {
  // Use non-square kernel to ensure dimension swapping is tested properly.
  const int64_t oh = 6, ow = 12, kh = 3, kw = 5;

  // Create Conv2DOp where the standard loop order is (oh, ow, kh, kw).
  OpBuilder builder(ctx.get());
  linalg::Conv2DOp conv2DOp = createConv2DOp(builder, oh, ow, kh, kw);
  FailureOr<ConvolutionDimensions> origDims = inferConvolutionDims(conv2DOp);
  ASSERT_TRUE(succeeded(origDims));
  ASSERT_EQ(origDims->outputImage.size(), 2u);
  ASSERT_EQ(origDims->filterLoop.size(), 2u);

  // Standard pairing: outputImage=[0,1], filterLoop=[2,3]
  // d0 <-> d2 (oh <-> kh), d1 <-> d3 (ow <-> kw)
  EXPECT_EQ(origDims->outputImage[0], 0u);
  EXPECT_EQ(origDims->outputImage[1], 1u);
  EXPECT_EQ(origDims->filterLoop[0], 2u);
  EXPECT_EQ(origDims->filterLoop[1], 3u);

  // Create equivalent generic with swapped filter loop order: (oh, ow, kw, kh)
  linalg::GenericOp swappedOp =
      createConv2DWithSwappedFilterLoops(builder, conv2DOp);
  FailureOr<ConvolutionDimensions> swappedDims =
      inferConvolutionDims(swappedOp);
  ASSERT_TRUE(succeeded(swappedDims));
  ASSERT_EQ(swappedDims->outputImage.size(), 2u);
  ASSERT_EQ(swappedDims->filterLoop.size(), 2u);

  // outputImage should still be [0, 1] after sorting.
  EXPECT_EQ(swappedDims->outputImage[0], 0u);
  EXPECT_EQ(swappedDims->outputImage[1], 1u);

  // In swapped version:
  //   Input map: (d0 + d3, d1 + d2) -> d0 <-> d3, d1 <-> d2
  // So filterLoop should be [3, 2] to maintain
  // outputImage[i] <-> filterLoop[i].
  EXPECT_EQ(swappedDims->filterLoop[0], 3u)
      << "outputImage[0]=0 should pair with filterLoop[0]=3 (oh <-> kh)";
  EXPECT_EQ(swappedDims->filterLoop[1], 2u)
      << "outputImage[1]=1 should pair with filterLoop[1]=2 (ow <-> kw)";
}

} // namespace
