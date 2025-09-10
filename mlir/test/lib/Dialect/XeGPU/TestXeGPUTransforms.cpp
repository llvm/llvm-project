//===- TestXeGPUTransforms.cpp -- Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xegpu;

namespace {

#define DEBUG_TYPE "test-xegpu-unroll"

struct TestXeGPUUnrollingPatterns
    : public PassWrapper<TestXeGPUUnrollingPatterns,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUUnrollingPatterns)

  StringRef getArgument() const final {
    return "test-xegpu-unrolling-patterns";
  }

  StringRef getDescription() const final {
    return "Test lowering patterns to unroll ops in the xegpu dialect";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
  }

  TestXeGPUUnrollingPatterns() = default;
  TestXeGPUUnrollingPatterns(const TestXeGPUUnrollingPatterns &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    xegpu::UnrollOptions options;
    options.setNativeShapeFn(
        [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
          if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp,
                  xegpu::PrefetchNdOp, xegpu::LoadNdOp, xegpu::StoreNdOp,
                  xegpu::CreateDescOp, xegpu::UpdateOffsetOp, xegpu::PrefetchOp,
                  xegpu::LoadGatherOp, xegpu::StoreScatterOp>(op)) {
            xegpu::TensorDescType tdescTy;
            if (auto createNdOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
              tdescTy = createNdOp.getType();
            } else if (auto updateNdOp =
                           dyn_cast<xegpu::UpdateNdOffsetOp>(op)) {
              tdescTy = updateNdOp.getTensorDescType();
            } else if (auto prefetchNdOp = dyn_cast<xegpu::PrefetchNdOp>(op)) {
              tdescTy = prefetchNdOp.getTensorDescType();
            } else if (auto loadNdOp = dyn_cast<xegpu::LoadNdOp>(op)) {
              tdescTy = loadNdOp.getTensorDescType();
            } else if (auto storeNdOp = dyn_cast<xegpu::StoreNdOp>(op)) {
              tdescTy = storeNdOp.getTensorDescType();
            } else if (auto createOp = dyn_cast<xegpu::CreateDescOp>(op)) {
              tdescTy = createOp.getType();
            } else if (auto updateOp = dyn_cast<xegpu::UpdateOffsetOp>(op)) {
              tdescTy = updateOp.getTensorDescType();
            } else if (auto prefetchOp = dyn_cast<xegpu::PrefetchOp>(op)) {
              tdescTy = prefetchOp.getTensorDescType();
            } else if (auto loadOp = dyn_cast<xegpu::LoadGatherOp>(op)) {
              tdescTy = loadOp.getTensorDescType();
            } else if (auto storeOp = dyn_cast<xegpu::StoreScatterOp>(op)) {
              tdescTy = storeOp.getTensorDescType();
            }

            if (auto layout = tdescTy.getLayoutAttr()) {
              auto inst_data = layout.getInstData();
              if (inst_data && layout.isForSubgroup())
                return SmallVector<int64_t>(inst_data.asArrayRef().begin(),
                                            inst_data.asArrayRef().end());
            }
          }

          if (isa<xegpu::DpasOp>(op))
            return SmallVector<int64_t>{8, 16, 16};

          return std::nullopt;
        });

    options.setUnrolledTypesFn(
        [&](ShapedType type, ArrayRef<int64_t> tileShape) -> SmallVector<Type> {
          Type elemTy = type.getElementType();
          Type newTy;

          // TensorDescType needs to drop the inst_data field in the layout
          // attribute
          if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type)) {
            Attribute encoding = tdescTy.getEncoding();
            auto layout = tdescTy.getLayoutAttr();

            // If the encoding is a ScatterTensorDescAttr, we need to
            // potentially adjust the chunk size based on the inst_data.
            if (tdescTy.isScattered()) {
              int64_t chunkSize = tdescTy.getChunkSizeAsInt();

              if (chunkSize > 1) {
                int64_t blockedChunkSize = chunkSize;
                auto instData = layout.getInstData();
                if (!instData.empty())
                  blockedChunkSize = instData.asArrayRef().back();

                // To create a new attribute with a different chunk_size:
                auto newEncoding = xegpu::ScatterTensorDescAttr::get(
                    ctx, tdescTy.getMemorySpace(), blockedChunkSize);

                encoding = newEncoding;
              }
            }
            if (layout) {
              if (layout.getLaneLayout() == nullptr)
                layout = xegpu::LayoutAttr();
              else
                layout = layout.dropInstData();
            }

            newTy = xegpu::TensorDescType::get(ctx, tileShape, elemTy, encoding,
                                               layout);

          } else {
            newTy = type.clone(tileShape, elemTy);
          }

          std::optional<SmallVector<int64_t>> ratio =
              computeShapeRatio(type.getShape(), tileShape);
          assert(ratio && "Expecting the ratio to be valid.");
          return SmallVector<Type>(computeProduct(*ratio), newTy);
        });

    RewritePatternSet patterns(ctx);

    populateXeGPUUnrollPatterns(patterns, options);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

#undef DEBUG_TYPE
#define DEBUG_TYPE "test-xegpu-layout-interface"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Test pattern for distributing vector::StepOp from workgroup to subgroup.
// Validates DistributeLayoutAttr interfaces for offset computation
// abstraction between LayoutAttr and SliceAttr.
class TestStepOpPattern : public OpConversionPattern<vector::StepOp> {
  using OpConversionPattern<vector::StepOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StepOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto layoutName = xegpu::getLayoutName(op->getResult(0));
    auto sliceAttr = op->getAttrOfType<xegpu::SliceAttr>(layoutName);
    if (!sliceAttr || sliceAttr.getRank() != 1)
      return failure();

    std::optional<SmallVector<int64_t>> sgShape = sliceAttr.getSgDataAsInt();
    if (!sgShape)
      return failure();

    Location loc = op.getLoc();
    VectorType type = op.getResult().getType();
    auto wgShape = type.getShape();

    Value sgId =
        gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
    auto maybeOffsets = sliceAttr.getOffsets(rewriter, loc, sgId, wgShape);
    if (failed(maybeOffsets))
      return failure();

    VectorType newTy = type.cloneWith(*sgShape, type.getElementType());
    Value base = vector::StepOp::create(rewriter, loc, newTy);
    SmallVector<Value> newOps;
    for (auto offsets : *maybeOffsets) {
      Value bcast =
          vector::BroadcastOp::create(rewriter, loc, newTy, offsets[0]);
      Value add = arith::AddIOp::create(rewriter, loc, base, bcast);
      newOps.push_back(add);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});
    return success();
  }
};

struct TestXeGPULayoutInterface
    : public PassWrapper<TestXeGPULayoutInterface,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPULayoutInterface)

  StringRef getArgument() const final { return "test-xegpu-layout-interface"; }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU Layout interfaces";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<index::IndexDialect>();
  }

  TestXeGPULayoutInterface() = default;
  TestXeGPULayoutInterface(const TestXeGPULayoutInterface &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    TypeConverter typeConverter;
    auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    RewritePatternSet patterns(ctx);
    patterns.add<TestStepOpPattern>(typeConverter, ctx);

    ConversionTarget target(*ctx);
    auto isLegal = [&](xegpu::SliceAttr layout) -> bool {
      return !layout || !layout.isForWorkgroup();
    };

    target.addDynamicallyLegalOp<vector::StepOp>(
        [&](vector::StepOp op) -> bool {
          auto layoutName = xegpu::getLayoutName(op->getResult(0));
          auto sliceAttr = op->getAttrOfType<xegpu::SliceAttr>(layoutName);
          return isLegal(sliceAttr);
        });

    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestXeGPULowerings() {
  PassRegistration<TestXeGPUUnrollingPatterns>();
  PassRegistration<TestXeGPULayoutInterface>();
}
} // namespace test
} // namespace mlir
