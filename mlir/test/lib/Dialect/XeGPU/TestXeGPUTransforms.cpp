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
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
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
      if (isa<xegpu::CreateNdDescOp, xegpu::PrefetchNdOp, xegpu::LoadNdOp,
              xegpu::StoreNdOp, xegpu::PrefetchOp, xegpu::LoadGatherOp,
              xegpu::StoreScatterOp>(op)) {
        xegpu::TensorDescType tdescTy;
        if (auto createNdOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
          tdescTy = createNdOp.getType();
        } else if (auto prefetchNdOp = dyn_cast<xegpu::PrefetchNdOp>(op)) {
          tdescTy = prefetchNdOp.getTensorDescType();
        } else if (auto loadNdOp = dyn_cast<xegpu::LoadNdOp>(op)) {
          tdescTy = loadNdOp.getTensorDescType();
        } else if (auto storeNdOp = dyn_cast<xegpu::StoreNdOp>(op)) {
          tdescTy = storeNdOp.getTensorDescType();
        } else if (isa<xegpu::PrefetchOp, xegpu::LoadGatherOp,
                       xegpu::StoreScatterOp>(op)) {
          auto anchorOp = cast<xegpu::AnchorLayoutInterface>(op);
          auto layout =
              dyn_cast_or_null<xegpu::LayoutAttr>(anchorOp.getAnchorLayout());
          if (layout && layout.isForSubgroup()) {
            auto inst_data = layout.getEffectiveInstDataAsInt();
            if (!inst_data.empty())
              return SmallVector<int64_t>(inst_data.begin(), inst_data.end());
          }
          return std::nullopt;
        }

        if (auto layout = tdescTy.getLayoutAttr()) {
          auto inst_data = layout.getEffectiveInstDataAsInt();
          if (!inst_data.empty() && layout.isForSubgroup())
            return SmallVector<int64_t>(inst_data.begin(), inst_data.end());
        }
      }

      if (isa<xegpu::DpasOp>(op))
        return SmallVector<int64_t>{8, 16, 16};

      // For vector.multi_reduction, read tile shape from the layout attribute
      // on the source operand (layout_operand_0).
      if (isa<vector::MultiDimReductionOp>(op)) {
        xegpu::DistributeLayoutAttr layout =
            xegpu::getDistributeLayoutAttr(op->getOpOperand(0));
        if (layout) {
          auto instData = layout.getEffectiveInstDataAsInt();
          if (!instData.empty())
            return instData;
        }
        return std::nullopt;
      }

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

            if (layout) {
              if (layout.getEffectiveLaneLayoutAsInt().empty())
                layout = xegpu::LayoutAttr();
              else
                layout = layout.dropInstData();
            }

            newTy = xegpu::TensorDescType::get(ctx, tileShape, elemTy, encoding,
                                               layout);
            // compute the product of batch (higher) dimensions
            ArrayRef<int64_t> shape = type.getShape();
            int64_t batchCount =
                shape.size() > 2 ? computeProduct(shape.drop_back(2)) : 1;
            return SmallVector<Type>(batchCount, newTy);
          }

          newTy = type.clone(tileShape, elemTy);
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

struct TestXeGPURecoverTemporaryLayouts
    : public PassWrapper<TestXeGPURecoverTemporaryLayouts,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPURecoverTemporaryLayouts)

  StringRef getArgument() const final {
    return "test-xegpu-recover-temporary-layouts";
  }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU temporary layout recovery";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPURecoverTemporaryLayouts() = default;
  TestXeGPURecoverTemporaryLayouts(const TestXeGPURecoverTemporaryLayouts &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    Operation *op = getOperation();
    if (!xegpu::recoverTemporaryLayouts(op))
      signalPassFailure();
  }
};

/// This test pass is intended to test the subgroup to lane distribution of
/// xegpu/vector/arith operations in isolation, it does not handle any
/// structural ops like scf.for etc.
struct TestXeGPUSgToLaneDistribute
    : public PassWrapper<TestXeGPUSgToLaneDistribute,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUSgToLaneDistribute)

  StringRef getArgument() const final {
    return "test-xegpu-sg-to-lane-distribute";
  }

  StringRef getDescription() const final {
    return "Test the implementation of XeGPU Subgroup to Lane Distribution";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<index::IndexDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  TestXeGPUSgToLaneDistribute() = default;
  TestXeGPUSgToLaneDistribute(const TestXeGPUSgToLaneDistribute &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    Operation *op = getOperation();
    if (!xegpu::recoverTemporaryLayouts(op)) {
      signalPassFailure();
      return;
    }

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

    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    xegpu::populateXeGPUSgToLaneDistributeTypeConversionAndLegality(
        typeConverter, patterns, target, op);
    (void)applyPartialConversion(op, target, std::move(patterns));
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
    if (failed(xegpu::propagateLayouts(builder, getOperation(), kind, 32))) {
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
    if (failed(xegpu::resolveLayoutConflicts(getOperation())))
      signalPassFailure();
  }
};

struct TestXeGPUArrayLengthOptimization
    : public PassWrapper<TestXeGPUArrayLengthOptimization,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUArrayLengthOptimization)

  StringRef getArgument() const final {
    return "test-xegpu-array-length-optimization";
  }

  StringRef getDescription() const final {
    return "Test XeGPU 2D block array load optimization patterns in isolation";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
  }

  TestXeGPUArrayLengthOptimization() = default;
  TestXeGPUArrayLengthOptimization(const TestXeGPUArrayLengthOptimization &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xegpu::populateXeGPUArrayLengthOptimizationPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
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

struct TestXeGPUCoalesceGatherScatter
    : public PassWrapper<TestXeGPUCoalesceGatherScatter, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUCoalesceGatherScatter)

  StringRef getArgument() const final {
    return "test-xegpu-coalesce-gather-scatter";
  }

  StringRef getDescription() const final {
    return "Test the XeGPU contiguity analysis and its coalescing consumer.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<xegpu::XeGPUDialect>();
  }

  TestXeGPUCoalesceGatherScatter() = default;
  TestXeGPUCoalesceGatherScatter(const TestXeGPUCoalesceGatherScatter &pass)
      : PassWrapper(pass) {}

  Option<unsigned> maxChunkSize{
      *this, "max-chunk-size",
      llvm::cl::desc("Upper bound on the produced lane_data FCD."),
      llvm::cl::init(8)};

  Option<bool> analyzeOnly{
      *this, "analyze-only",
      llvm::cl::desc("Only run the analysis (stamp contiguous_chunk "
                     "attributes); do not apply."),
      llvm::cl::init(false)};

  void runOnOperation() override {
    xegpu::runCoalesceGatherScatterAnalysis(getOperation());
    if (analyzeOnly)
      return;
    getOperation()->walk([&](Operation *op) {
      if (auto load = dyn_cast<xegpu::LoadGatherOp>(op))
        applyContiguousChunk(load, maxChunkSize);
      else if (auto store = dyn_cast<xegpu::StoreScatterOp>(op))
        applyContiguousChunk(store, maxChunkSize);
    });
  }

private:
  /// Largest power-of-two `<= bound` that divides `numLanes`.
  static int64_t largestPow2Divisor(int64_t numLanes, int64_t bound) {
    if (bound < 2 || numLanes < 2)
      return 1;
    int64_t f = std::min<int64_t>(bound, numLanes);
    // Round down to power of 2.
    if (!llvm::isPowerOf2_64(f))
      f = static_cast<int64_t>(llvm::bit_floor(static_cast<uint64_t>(f)));
    while (f >= 2) {
      if (numLanes % f == 0)
        return f;
      f /= 2;
    }
    return 1;
  }

  /// Look up the subgroup size from the enclosing gpu.module's xevm.target.
  /// Falls back to 16 when no target chip is found or the chip is unknown,
  /// matching the typical Intel Xe2 default.
  static unsigned lookupSubgroupSize(Operation *op) {
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    return uArch ? static_cast<unsigned>(uArch->getSubgroupSize()) : 16u;
  }

  /// Build a `lane_layout`/`lane_data`/`inst_data` layout of rank `rank`, with
  /// the given lane_layout / lane_data on the innermost dim (1 elsewhere).
  /// `inst_data` is `lane_layout * lane_data` per dim.
  static xegpu::LayoutAttr buildLaneDataLayout(MLIRContext *ctx, unsigned rank,
                                               int64_t innerLaneLayout,
                                               int64_t innerLaneData) {
    SmallVector<int32_t> laneLayout(rank, 1);
    SmallVector<int32_t> laneData(rank, 1);
    SmallVector<int32_t> instData(rank, 1);
    laneLayout.back() = static_cast<int32_t>(innerLaneLayout);
    laneData.back() = static_cast<int32_t>(innerLaneData);
    instData.back() = static_cast<int32_t>(innerLaneLayout * innerLaneData);
    return xegpu::LayoutAttr::get(ctx, instData, laneLayout, laneData);
  }

  /// Returns true if `mask` is a constant `dense<true>` vector.
  static bool isAllTrueMask(Value mask) {
    auto vecTy = dyn_cast<VectorType>(mask.getType());
    if (!vecTy)
      return false;
    auto cst = mask.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return false;
    auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return false;
    return dense.getSplatValue<APInt>().getBoolValue();
  }

  /// True when `op` (a gather load or scatter store) is tied to a
  /// `vector.multi_reduction`: a load whose result feeds a reduction, or a
  /// store whose stored value comes from one (through layout-neutral /
  /// elementwise / insert glue). Coalescing such an access is gated off here;
  /// it requires reduction-aware layout handling added in follow-up PRs.
  template <typename OpTy>
  static bool isReductionTied(OpTy op) {
    if constexpr (std::is_same_v<OpTy, xegpu::StoreScatterOp>) {
      SmallVector<Value, 8> worklist{op.getValue()};
      llvm::SmallPtrSet<Operation *, 16> seen;
      unsigned steps = 0;
      while (!worklist.empty() && steps++ < 64) {
        Value v = worklist.pop_back_val();
        Operation *def = v.getDefiningOp();
        if (!def || !seen.insert(def).second)
          continue;
        if (isa<vector::MultiDimReductionOp>(def))
          return true;
        if (isa<vector::ShapeCastOp, vector::BitCastOp, xegpu::ConvertLayoutOp,
                vector::InsertOp, vector::InsertStridedSliceOp>(def) ||
            OpTrait::hasElementwiseMappableTraits(def))
          for (Value operand : def->getOperands())
            if (isa<VectorType>(operand.getType()))
              worklist.push_back(operand);
      }
      return false;
    } else {
      for (Operation *user : op->getUsers()) {
        Operation *u = user;
        while (
            u &&
            isa<vector::ShapeCastOp, vector::BitCastOp, xegpu::ConvertLayoutOp>(
                u)) {
          if (u->getNumResults() != 1 || u->getResult(0).use_empty())
            break;
          u = *u->getResult(0).getUsers().begin();
        }
        if (u && isa<vector::MultiDimReductionOp>(u))
          return true;
      }
      return false;
    }
  }

  /// Turn a stamped `contiguous_chunk` on `op` into a lane_layout / lane_data /
  /// inst_data layout, capped by `maxChunkSize`, then remove the attribute.
  /// Skips ops with a non-uniform mask, an existing lane_data, or a reduction
  /// tie — these are coalescing concerns, not properties of the offsets.
  template <typename OpTy>
  static void applyContiguousChunk(OpTy op, unsigned maxChunkSize) {
    std::optional<uint64_t> chunk = op.getContiguousChunk();
    if (!chunk)
      return;
    auto cleanup = llvm::scope_exit([&] { op.removeContiguousChunkAttr(); });

    auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
    auto valueTy = op.getValueType();
    if (!offsetsTy || !valueTy || offsetsTy.getNumElements() <= 1)
      return;
    if (!isAllTrueMask(op.getMask()))
      return;
    if (auto layout = op.getLayoutAttr())
      if (!layout.getEffectiveLaneDataAsInt().empty())
        return;
    if (isReductionTied(op))
      return;

    int64_t inner = offsetsTy.getShape().back();
    unsigned subgroupSize = lookupSubgroupSize(op);
    // lane_layout default: min(subgroupSize, inner) rounded to a divisor.
    int64_t laneLayout =
        largestPow2Divisor(inner, std::min<int64_t>(subgroupSize, inner));
    if (laneLayout < 1)
      return;
    int64_t perLane = inner / laneLayout;
    // lane_data = min(contiguity, maxChunkSize, perLane), pow2 divisor of
    // perLane.
    int64_t bound =
        std::min<int64_t>({static_cast<int64_t>(*chunk),
                           static_cast<int64_t>(maxChunkSize), perLane});
    int64_t factor = largestPow2Divisor(perLane, bound);
    if (factor < 2)
      return;

    op.setLayoutAttr(buildLaneDataLayout(op.getContext(), valueTy.getRank(),
                                         laneLayout, factor));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestXeGPULowerings() {
  PassRegistration<TestXeGPUUnrollingPatterns>();
  PassRegistration<TestXeGPULayoutInterface>();
  PassRegistration<TestXeGPURecoverTemporaryLayouts>();
  PassRegistration<TestXeGPUSgToLaneDistribute>();
  PassRegistration<TestXeGPUPropagateLayouts>();
  PassRegistration<TestXeGPUResolveLayoutConflicts>();
  PassRegistration<TestXeGPUArrayLengthOptimization>();
  PassRegistration<TestXeGPUCoalesceGatherScatter>();
}
} // namespace test
} // namespace mlir
