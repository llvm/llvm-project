//===- TestXeGPUTransforms.cpp -- Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

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
  TestXeGPUUnrollingPatterns(const TestXeGPUUnrollingPatterns &pass) = default;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    xegpu::UnrollOptions options;
    options.setNativeShapeFn([&](Operation *op)
                                 -> std::optional<SmallVector<int64_t>> {
      if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp,
              xegpu::PrefetchNdOp, xegpu::LoadNdOp, xegpu::StoreNdOp,
              xegpu::CreateDescOp, xegpu::UpdateOffsetOp, xegpu::PrefetchOp,
              xegpu::LoadGatherOp, xegpu::StoreScatterOp>(op)) {
        xegpu::TensorDescType tdescTy;
        if (auto createNdOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
          tdescTy = createNdOp.getType();
        } else if (auto updateNdOp = dyn_cast<xegpu::UpdateNdOffsetOp>(op)) {
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
          if (loadOp.getOffsets()) {
            auto layout = xegpu::getDistributeLayoutAttr(loadOp.getResult());
            if (layout && layout.isForSubgroup()) {
              auto inst_data = layout.getEffectiveInstDataAsInt();
              if (!inst_data.empty())
                return SmallVector<int64_t>(inst_data.begin(), inst_data.end());
            }
            return std::nullopt;
          }
          tdescTy = loadOp.getTensorDescType();
        } else if (auto storeOp = dyn_cast<xegpu::StoreScatterOp>(op)) {
          if (storeOp.getOffsets()) {
            auto layout = llvm::dyn_cast_or_null<xegpu::LayoutAttr>(
                op->getAttr("layout"));
            if (layout && layout.isForSubgroup()) {
              auto inst_data = layout.getEffectiveInstDataAsInt();
              if (!inst_data.empty())
                return SmallVector<int64_t>(inst_data.begin(), inst_data.end());
            }
            return std::nullopt;
          }
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
        [&](ShapedType type, ArrayRef<int64_t> tileShape,
            bool returnSingleType = false) -> SmallVector<Type> {
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

          if (returnSingleType)
            return SmallVector<Type>{newTy};
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

// Test pattern for distributing vector::StepOp from workgroup to subgroup.
// Validates DistributeLayoutAttr interfaces for offset computation
// abstraction between LayoutAttr and SliceAttr.
class TestStepOpPattern : public OpConversionPattern<vector::StepOp> {
  using OpConversionPattern<vector::StepOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StepOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto layoutName = xegpu::getTemporaryLayoutName(op->getResult(0));
    auto sliceAttr = op->getAttrOfType<xegpu::SliceAttr>(layoutName);
    if (!sliceAttr || sliceAttr.getRank() != 1)
      return failure();

    std::optional<SmallVector<int64_t>> sgShape =
        sliceAttr.getEffectiveSgDataAsInt();
    if (!sgShape)
      return failure();

    Location loc = op.getLoc();
    VectorType type = op.getResult().getType();
    auto wgShape = type.getShape();

    Value sgId =
        gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
    auto maybeOffsets =
        sliceAttr.computeDistributedCoords(rewriter, loc, sgId, wgShape);
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

struct TestXeGPUSGDistribute
    : public PassWrapper<TestXeGPUSGDistribute,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUSGDistribute)

  StringRef getArgument() const final { return "test-xegpu-sg-distribute"; }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU Subgroup Distribution";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<index::IndexDialect>();
  }

  TestXeGPUSGDistribute() = default;
  TestXeGPUSGDistribute(const TestXeGPUSGDistribute &pass) = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xegpu::populateXeGPUSubgroupDistributePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

/// This test pass is intended to test the subgroup to workitem distribution of
/// xegpu/vector/arith operations in isolation, it does not handle any
/// structural ops like scf.for etc.
struct TestXeGPUSgToWiDistributeExperimental
    : public PassWrapper<TestXeGPUSgToWiDistributeExperimental,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestXeGPUSgToWiDistributeExperimental)

  StringRef getArgument() const final {
    return "test-xegpu-sg-to-wi-distribute-experimental";
  }

  StringRef getDescription() const final {
    return "Test the experimental implementation of XeGPU Subgroup to "
           "Work-item Distribution";
  }

  Option<bool> enableRewriteMultiReductionToReductions{
      *this, "enable-rewrite-multi-reduction-to-reductions",
      llvm::cl::desc("Partially lower multi-reduction ops to reduction ops if "
                     "the reduction dimension is distributed."),
      llvm::cl::init(false)};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<index::IndexDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPUSgToWiDistributeExperimental() = default;
  TestXeGPUSgToWiDistributeExperimental(
      const TestXeGPUSgToWiDistributeExperimental &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    TypeConverter typeConverter;
    // Define type materializations using UnrealizedConversionCastOp.
    auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // If `enableRewriteMultiReductionToReductions` is set, only focus on
    // testing the partial lowering of vector::MultiReductionOp.
    if (enableRewriteMultiReductionToReductions) {
      xegpu::populateXeGPUSgToWiDistributeTypeConversions(typeConverter);
      ConversionTarget target(*ctx);
      RewritePatternSet patterns(ctx);
      xegpu::populateXeGPUSgToWiLowerVectorMultiReductionAndLegality(patterns,
                                                                     target);
      (void)applyPartialConversion(getOperation(), target, std::move(patterns));
      return;
    }

    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    xegpu::populateXeGPUSgToWiDistributeTypeConversionAndLegality(
        typeConverter, patterns, target);
    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct TestXeGPUMoveFuncBodyToWarpOp
    : public PassWrapper<TestXeGPUMoveFuncBodyToWarpOp,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUMoveFuncBodyToWarpOp)

  StringRef getArgument() const final {
    return "test-xegpu-move-func-to-warp-op";
  }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU move gpu function body to "
           "WarpExecuteOnLane0 op.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPUMoveFuncBodyToWarpOp() = default;
  TestXeGPUMoveFuncBodyToWarpOp(const TestXeGPUMoveFuncBodyToWarpOp &pass) =
      default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xegpu::populateXeGPUMoveFuncBodyToWarpOpPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

struct TestXeGPUPropagateLayouts
    : public PassWrapper<TestXeGPUPropagateLayouts,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUPropagateLayouts)

  StringRef getArgument() const final { return "test-xegpu-propagate-layouts"; }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU propagate layouts.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPUPropagateLayouts() = default;
  TestXeGPUPropagateLayouts(const TestXeGPUPropagateLayouts &pass)
      : PassWrapper(pass) {}

  Option<std::string> layoutKind{
      *this, "layout-kind",
      llvm::cl::desc("Propagate `subgroup` / `inst` / `lane` level of xegpu "
                     "layouts."),
      llvm::cl::init("lane")};

  void runOnOperation() override {
    OpBuilder builder(getOperation());
    LayoutKind kind;
    if (layoutKind == "subgroup")
      kind = LayoutKind::Subgroup;
    else if (layoutKind == "inst")
      kind = LayoutKind::InstData;
    else if (layoutKind == "lane")
      kind = LayoutKind::Lane;
    else {
      signalPassFailure();
      return;
    }
    if (failed(xegpu::propagateLayouts(builder, getOperation(), kind))) {
      signalPassFailure();
    }
  }
};

struct TestXeGPUResolveLayoutConflicts
    : public PassWrapper<TestXeGPUResolveLayoutConflicts,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUResolveLayoutConflicts)

  StringRef getArgument() const final {
    return "test-xegpu-resolve-layout-conflicts";
  }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU layout conflict resolution.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPUResolveLayoutConflicts() = default;
  TestXeGPUResolveLayoutConflicts(const TestXeGPUResolveLayoutConflicts &pass) =
      default;

  void runOnOperation() override {
    if (failed(xegpu::resolveLayoutConflicts(getOperation()))) {
      signalPassFailure();
    }
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
  TestXeGPULayoutInterface(const TestXeGPULayoutInterface &pass) = default;

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
          auto layoutName = xegpu::getTemporaryLayoutName(op->getResult(0));
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
  PassRegistration<TestXeGPUSGDistribute>();
  PassRegistration<TestXeGPUSgToWiDistributeExperimental>();
  PassRegistration<TestXeGPUMoveFuncBodyToWarpOp>();
  PassRegistration<TestXeGPUPropagateLayouts>();
  PassRegistration<TestXeGPUResolveLayoutConflicts>();
}
} // namespace test
} // namespace mlir
