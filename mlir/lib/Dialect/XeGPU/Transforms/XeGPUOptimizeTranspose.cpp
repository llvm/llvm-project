//===- XeGPUOptimizeTranspose.cpp - XeGPU optimize transpose ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUOPTIMIZETRANSPOSE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-optimize-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

static std::optional<SmallVector<int64_t>>
get2DLaneData(xegpu::TensorDescType tdescType) {
  auto layout = tdescType.getLayoutAttr();
  if (!layout)
    return std::nullopt;
  auto laneData = layout.getEffectiveLaneDataAsInt();
  if (laneData.size() != 2)
    return std::nullopt;
  return laneData;
}

class XeGPUCreateNdDescOpPattern final
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
public:
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp createNdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdescTy = createNdOp.getType();
    auto convertType = this->getTypeConverter()->convertType(tdescTy);
    if (convertType == tdescTy)
      return failure();
    return success();
  }
};
} // namespace

void xegpu::populateXeGPUOptimizeTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<XeGPUCreateNdDescOpPattern>(patterns.getContext());
}

namespace {

struct XeGPUOptimizeTransposePass final
    : public xegpu::impl::XeGPUOptimizeTransposeBase<
          XeGPUOptimizeTransposePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
        [](xegpu::CreateNdDescOp createNdOp) {
          auto optionalLaneData = get2DLaneData(createNdOp.getType());
          if (!optionalLaneData)
            return true;
          auto laneData = optionalLaneData.value();
          return laneData[0] != 1 || laneData[1] == 1;
        });

    converter.addConversion([](xegpu::TensorDescType tdescType) {
      auto optionalLaneData = get2DLaneData(tdescType);
      if (!optionalLaneData)
        return tdescType;
      auto laneData = optionalLaneData.value();
      int64_t innerLaneData = laneData[1];
      if (laneData[0] == 1 && innerLaneData != 1) {
        int elementTyBitwidth =
            tdescType.getElementType().getIntOrFloatBitWidth();
        assert(elementTyBitwidth < 32 &&
               "Expected element type bitwidth < 32 with laneData[1] != 1");
        SmallVector<int64_t> newShape(tdescType.getShape());
        newShape.back() = newShape.back() / innerLaneData;
        Type newElemTy = IntegerType::get(tdescType.getContext(),
                                          elementTyBitwidth * innerLaneData);
        xegpu::LayoutAttr newLayout = xegpu::LayoutAttr::get(
            tdescType.getContext(),
            tdescType.getLayoutAttr().getLaneLayout().asArrayRef(), {1, 1});
        return xegpu::TensorDescType::get(
            newShape, newElemTy, tdescType.getArrayLength(),
            tdescType.getBoundaryCheck(), tdescType.getMemorySpace(),
            newLayout);
      }
      return tdescType;
    });

    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    xegpu::populateXeGPUOptimizeTransposePatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      DBGS() << "Optimize transpose pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
