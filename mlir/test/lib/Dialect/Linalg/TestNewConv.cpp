//===- TestNewConv.cpp - Test `linalg.conv` -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass which converts "old" convolution ops, e.g.
// `linalg.depthwise_conv_2d_nhwc`, `linalg.conv_2d_nhwc`, etc., to the new
// `linalg.conv` op.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
class OldToNewConv : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::isa<linalg::ConvOp>(op))
      return failure();
    auto nameStr = op->getName().stripDialect().str();

    bool isDepthwise = nameStr.substr(0, 14) == "depthwise_conv";
    if (isDepthwise)
      nameStr = nameStr.substr(15);
    else if (nameStr.substr(0, 4) == "conv")
      nameStr = nameStr.substr(5);
    else
      return failure();

    int64_t spatialDims;
    {
      auto dimensionality = nameStr.substr(0, 2);
      if (dimensionality == "1d")
        spatialDims = 1;
      else if (dimensionality == "2d")
        spatialDims = 2;
      else if (dimensionality == "3d")
        spatialDims = 3;
      else
        return failure();
    }

    SmallVector<ConvDimEnum, 4> inputDims, filterDims, outputDims;
    if (nameStr.length() == 2) {

      // These are the ops `conv_1d`, `conv_2d` and `conv_3d` which use only
      // spatial dimensions.
      if (spatialDims == 1)
        filterDims = inputDims = {ConvDimEnum::SPATIAL_0};
      else if (spatialDims == 2)
        filterDims =
            inputDims = {ConvDimEnum::SPATIAL_0, ConvDimEnum::SPATIAL_1};
      else if (spatialDims == 3)
        filterDims =
            inputDims = {ConvDimEnum::SPATIAL_0, ConvDimEnum::SPATIAL_1,
                         ConvDimEnum::SPATIAL_2};
      else
        return failure();

    } else {
      // This handles all the ops with specialized tensor dimension orders like
      // `conv_2d_nhwc_fhwc`, `depthwise_conv_2d_nhwc_hwc`, etc.
      auto specialization = nameStr.substr(3); // get rid of first _

      // Separator between input and filter layout.
      auto sep = specialization.find('_');
      if (sep == StringRef::npos)
        return failure();
      auto inputDimStr = specialization.substr(0, sep);
      auto filterDimStr = specialization.substr(sep + 1);

      auto parseDim = [&](char c) -> ConvDimEnum {
        switch (c) {
        case 'n':
          return ConvDimEnum::BATCH;
        case 'h':
          return ConvDimEnum::SPATIAL_1;
        case 'w':
          return ConvDimEnum::SPATIAL_0;
        case 'd':
          return ConvDimEnum::SPATIAL_2;
        case 'f':
          return ConvDimEnum::OUTPUT_CHANNEL;
        case 'g':
          return ConvDimEnum::GROUP;
        case 'c':
          // The old convolution ops use the letter 'c' to denote a
          // non-reduction dimension in all tensors in the depthwise case. The
          // new convolution captures this behavior in the group dimension.
          return isDepthwise ? ConvDimEnum::GROUP : ConvDimEnum::INPUT_CHANNEL;
        case 'm':
          // Similarly, the old convolution ops use the letter 'm' to denote a
          // parallel dimesion in filter and output in the depthwise case. This
          // behavior is captured by the ordinary output channel dimension.
          assert(isDepthwise && "Unexpected letter 'm' in non-depthwise conv");
          return ConvDimEnum::OUTPUT_CHANNEL;
        default:
          llvm_unreachable("unknown dimensional character ");
        }
      };

      inputDims = llvm::map_to_vector(inputDimStr, parseDim);
      filterDims = llvm::map_to_vector(filterDimStr, parseDim);
    }

    // This is the behavior of the old convolution ops:
    // The output dimension order is the same as the input dimension order, but
    // output channel stands in for input channel...
    for (auto d : inputDims)
      if (d == ConvDimEnum::INPUT_CHANNEL)
        outputDims.push_back(ConvDimEnum::OUTPUT_CHANNEL);
      else
        outputDims.push_back(d);
    // ... and if the "depthwise channel multiplier" dimension 'm' appears, the
    // output tensor has an additional dimension appended.
    if (isDepthwise &&
        llvm::is_contained(filterDims, ConvDimEnum::OUTPUT_CHANNEL))
      outputDims.push_back(ConvDimEnum::OUTPUT_CHANNEL);

    SmallVector<int64_t> strides(spatialDims, 1), dilations(spatialDims, 1);
    // The old convolution ops order the strides and dilations in the order "D,
    // H, W". We order them as spatial 0, spatial 1, spatial 2, so we have to
    // reverse the order.
    if (op->hasAttr("strides"))
      strides = SmallVector<int64_t>(llvm::reverse(
          SmallVector<int64_t>(op->getAttrOfType<DenseElementsAttr>("strides")
                                   .getValues<int64_t>())));
    if (op->hasAttr("dilations"))
      dilations = SmallVector<int64_t>(llvm::reverse(
          SmallVector<int64_t>(op->getAttrOfType<DenseElementsAttr>("dilations")
                                   .getValues<int64_t>())));

    rewriter.replaceOpWithNewOp<linalg::ConvOp>(
        op, op->getResultTypes(), op->getOperand(0), op->getOperand(1),
        op->getOperand(2), inputDims, filterDims, outputDims, strides,
        dilations);

    return success();
  }
};

struct TestNewConvPass : public PassWrapper<TestNewConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNewConvPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  StringRef getArgument() const final { return "test-linalg-new-conv"; }
  StringRef getDescription() const final { return "Test new linalg.conv Op"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(getContext());

    target.addLegalOp<linalg::ConvOp>();
    // Every non-converted old conv op should fail the converison.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getStringRef().str().find("conv") ==
             std::string::npos;
    });

    patterns.add<OldToNewConv>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestNewConv() { PassRegistration<TestNewConvPass>(); }
} // namespace test
} // namespace mlir
