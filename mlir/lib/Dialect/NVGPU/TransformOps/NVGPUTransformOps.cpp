//===- NVGPUTransformOps.cpp - Implementation of NVGPU transform ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::nvgpu;
using namespace mlir::transform;

#define DEBUG_TYPE "nvgpu-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

//===----------------------------------------------------------------------===//
// RewriteMatmulAsMmaSyncOp
//===----------------------------------------------------------------------===//

/// Helper struct to encode a pair of row/column indexings in the form of
/// affine expressions.
struct RowColIndexing : private std::pair<AffineExpr, AffineExpr> {
  RowColIndexing(AffineExpr row, AffineExpr col)
      : std::pair<AffineExpr, AffineExpr>(row, col) {}

  AffineExpr row() const { return first; };
  AffineExpr col() const { return second; };

  void print(llvm::raw_ostream &os) const {
    os << "- indexing: " << first << ", " << second;
  }
};

/// Helper struct to provide a simple mapping from matmul operations to the
/// corresponding mma.sync operation. This is constrained to the case where the
/// matmul matches the mma.sync operation 1-1.
struct MmaSyncBuilder {
  MmaSyncBuilder(OpBuilder &b, Location loc, OpFoldResult laneId)
      : b(b), loc(loc), laneId(laneId) {}

  using IndexCalculator =
      std::function<SmallVector<RowColIndexing>(MLIRContext *)>;

  /// Create the mma.sync operation corresponding to `linalgOp` along with all
  /// the supporting load/store and vector operations.
  FailureOr<Operation *> buildMmaSync(LinalgOp linalgOp);

private:
  struct MmaSyncInfo {
    std::tuple<IndexCalculator, IndexCalculator, IndexCalculator> indexFns;
    std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>>
        vectorShapes;
    SmallVector<int64_t> mmaShape;
    bool tf32Enabled;
  };

  /// Return the specific index calculator for the given `linalgOp` or failure
  /// if the op is not supported. This is the toplevel switch that should just
  /// be Tablegen'd in the future.
  FailureOr<MmaSyncInfo> getIndexCalculators(ArrayRef<int64_t> opShape,
                                             TypeRange elementalTypes);

  //===--------------------------------------------------------------------===//
  // Instruction-specific row, column indexing expression builders.
  // These should all be declaratively specified via Tablegen in the future.
  // The Tablegen specification should be as straightforward as possible to
  // only model the existing size and type combinations.
  //===--------------------------------------------------------------------===//
  //
  // TODO: Tablegen all this.
  //===--------------------------------------------------------------------===//
  // m16n8k4 tf32 case.
  //===--------------------------------------------------------------------===//
  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =      groupID            for a0
  ///            groupID + 8        for a1
  /// col =  threadIDInGroup
  static SmallVector<RowColIndexing> m16n8k4tf32Lhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{groupID, threadIDInGroup},
            RowColIndexing{groupID + 8, threadIDInGroup}};
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =  threadIDInGroup
  /// col =  groupID
  static SmallVector<RowColIndexing> m16n8k4tf32Rhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{threadIDInGroup, groupID}};
  }

  /// From the NVIDIA doc:
  /// groupID          = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =      groupID                            for c0 and c1
  ///          groupID + 8                          for c2 and c3
  /// col =  (threadIDInGroup * 2) + (i & 0x1)    for ci   where i = {0,..,3}
  static SmallVector<RowColIndexing> m16n8k4tf32Res(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{groupID, threadIDInGroup * 2 + 0},
            RowColIndexing{groupID, threadIDInGroup * 2 + 1},
            RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},
            RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1}};
  }

  //===--------------------------------------------------------------------===//
  // m16n8k16 f16 case.
  //===--------------------------------------------------------------------===//
  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
  ///           groupID + 8         Otherwise
  ///
  /// col =  (threadIDInGroup * 2) + (i & 0x1)          for ai where i <  4
  ///        (threadIDInGroup * 2) + (i & 0x1) + 8      for ai where i >= 4
  static SmallVector<RowColIndexing> m16n8k16f16Lhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{groupID, threadIDInGroup * 2 + 0},         // i == 0
      RowColIndexing{groupID, threadIDInGroup * 2 + 1},         // i == 1
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},     // i == 2
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1},     // i == 3
      RowColIndexing{groupID, threadIDInGroup * 2 + 0 + 8},     // i == 4
      RowColIndexing{groupID, threadIDInGroup * 2 + 1 + 8},     // i == 5
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0 + 8}, // i == 6
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1 + 8}  // i == 7
    };
    // clang-format on
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =  (threadIDInGroup * 2) + (i & 0x1)           for bi where i <  2
  ///        (threadIDInGroup * 2) + (i & 0x1) + 8       for bi where i >= 2
  ///
  /// col = groupID
  static SmallVector<RowColIndexing> m16n8k16f16Rhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{threadIDInGroup * 2 + 0, groupID},        // i == 0
      RowColIndexing{threadIDInGroup * 2 + 1, groupID},        // i == 1
      RowColIndexing{threadIDInGroup * 2 + 0 + 8, groupID},    // i == 2
      RowColIndexing{threadIDInGroup * 2 + 1 + 8, groupID}     // i == 3
    };
    // clang-format on
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =      groupID                               for ci where i <  2
  ///          groupID + 8                             for ci where i >= 2
  ///
  /// col =  (threadIDInGroup * 2) + (i & 0x1)      for ci where i = {0,..,3}
  static SmallVector<RowColIndexing> m16n8k16f16Res(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{groupID, threadIDInGroup * 2 + 0},        // i == 0
      RowColIndexing{groupID, threadIDInGroup * 2 + 1},        // i == 1
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},    // i == 2
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1}     // i == 3
    };
    // clang-format on
  }

  //===--------------------------------------------------------------------===//
  /// Helper functions to create customizable load and stores operations. The
  /// specific shapes of each MMA instruction are passed via the
  /// IndexCalculator callback.
  //===--------------------------------------------------------------------===//
  /// Build a list of memref.load operations indexed at `(row, col)` indices
  /// that make sense for a particular MMA instruction and specified via the
  /// IndexCalculator callback.
  SmallVector<Value> buildMemrefLoads(OpBuilder &b, Location loc,
                                      OpFoldResult laneId, Value memref,
                                      IndexCalculator indexFn);

  /// Perform a distributed load of a vector operand of `vectorShape` for a
  /// particular MMA instruction whose `(row, col)` indices are specified via
  /// the IndexCalculator callback. Each `laneId` loads the subportion of the
  /// data that makes sense for the particular MMA operation.
  /// The `vectorShape` matches existing NVGPU dialect op specification but
  /// could also be flattened in the future if needed for simplification.
  Value buildMmaSyncMemrefLoadOperand(OpBuilder &b, Location loc,
                                      OpFoldResult laneId, Value memref,
                                      IndexCalculator indexFn,
                                      ArrayRef<int64_t> vectorShape);

  /// Build a list of memref.store operations indexed at `(row, col)` indices
  /// that make sense for a particular MMA instruction and specified via the
  /// IndexCalculator callback.
  SmallVector<Operation *> buildMemrefStores(OpBuilder &b, Location loc,
                                             ValueRange toStore,
                                             OpFoldResult laneId, Value memref,
                                             IndexCalculator indexFn);

  /// Perform a distributed store of a vector operand of `vectorShape` for a
  /// particular MMA instruction whose `(row, col)` indices are specified via
  /// the IndexCalculator callback. Each `laneId` loads the subportion of the
  /// data that makes sense for the particular MMA operation.
  /// The `vectorShape` matches existing NVGPU dialect op specification but
  /// could also be flattened in the future if needed for simplification.
  SmallVector<Operation *> buildMmaSyncMemrefStoreOperand(
      OpBuilder &b, Location loc, Value vectorToStore, OpFoldResult laneId,
      Value memref, IndexCalculator indexFn, ArrayRef<int64_t> vectorShape);

  OpBuilder &b;
  Location loc;
  OpFoldResult laneId;
};

//===--------------------------------------------------------------------===//
/// Helper functions to create customizable load and stores operations. The
/// specific shapes of each MMA instruction are passed via the
/// IndexCalculator callback.
//===--------------------------------------------------------------------===//

template <typename ApplyFn, typename ReduceFn>
static void foreachIndividualVectorElement(Value vector, ApplyFn applyFn,
                                           ReduceFn reduceFn) {
  VectorType vectorType = vector.getType().cast<VectorType>();
  auto vectorShape = vectorType.getShape();
  auto strides = computeStrides(vectorShape);
  for (int64_t idx = 0, e = vectorShape[0] * strides[0]; idx < e; ++idx) {
    auto indices = delinearize(idx, strides);
    reduceFn(applyFn(vector, idx, indices), idx, indices);
  }
}

SmallVector<Value> MmaSyncBuilder::buildMemrefLoads(OpBuilder &b, Location loc,
                                                    OpFoldResult laneId,
                                                    Value memref,
                                                    IndexCalculator indexFn) {
  auto aff = [&](AffineExpr e) {
    return affine::makeComposedFoldedAffineApply(b, loc, e, laneId);
  };
  SmallVector<Value> res;
  SmallVector<RowColIndexing> indexings = indexFn(b.getContext());
  for (auto indexing : indexings) {
    Value row = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.row()));
    Value col = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.col()));
    auto load = b.create<memref::LoadOp>(loc, memref, ValueRange{row, col});
    res.push_back(load);
  }
  return res;
}

Value MmaSyncBuilder::buildMmaSyncMemrefLoadOperand(
    OpBuilder &b, Location loc, OpFoldResult laneId, Value memref,
    IndexCalculator indexFn, ArrayRef<int64_t> vectorShape) {
  auto loads = buildMemrefLoads(b, loc, laneId, memref, indexFn);

  Type elementType = getElementTypeOrSelf(memref.getType());
  auto vt = VectorType::get(vectorShape, elementType);
  Value res = b.create<vector::SplatOp>(loc, vt, loads[0]);
  foreachIndividualVectorElement(
      res,
      /*applyFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        return loads[linearIdx];
      },
      /*reduceFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        res = b.create<vector::InsertOp>(loc, v, res, indices);
      });

  return res;
}

SmallVector<Operation *>
MmaSyncBuilder::buildMemrefStores(OpBuilder &b, Location loc,
                                  ValueRange toStore, OpFoldResult laneId,
                                  Value memref, IndexCalculator indexFn) {
  auto aff = [&](AffineExpr e) {
    return affine::makeComposedFoldedAffineApply(b, loc, e, laneId);
  };
  SmallVector<Operation *> res;
  for (auto [indexing, val] :
       llvm::zip_equal(indexFn(b.getContext()), toStore)) {
    Value row = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.row()));
    Value col = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.col()));
    Operation *store =
        b.create<memref::StoreOp>(loc, val, memref, ValueRange{row, col});
    res.push_back(store);
  }
  return res;
}

SmallVector<Operation *> MmaSyncBuilder::buildMmaSyncMemrefStoreOperand(
    OpBuilder &b, Location loc, Value vectorToStore, OpFoldResult laneId,
    Value memref, IndexCalculator indexFn, ArrayRef<int64_t> vectorShape) {
  SmallVector<Value> toStore;
  toStore.reserve(32);
  foreachIndividualVectorElement(
      vectorToStore,
      /*applyFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        return b.create<vector::ExtractOp>(loc, vectorToStore, indices);
      },
      /*reduceFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        toStore.push_back(v);
      });
  return buildMemrefStores(b, loc, toStore, laneId, memref, indexFn);
}

static std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                  SmallVector<int64_t>>
makeVectorShapes(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs,
                 ArrayRef<int64_t> res) {
  SmallVector<int64_t> vlhs{lhs.begin(), lhs.end()};
  SmallVector<int64_t> vrhs{rhs.begin(), rhs.end()};
  SmallVector<int64_t> vres{res.begin(), res.end()};
  return std::make_tuple(vlhs, vrhs, vres);
}

FailureOr<MmaSyncBuilder::MmaSyncInfo>
MmaSyncBuilder::getIndexCalculators(ArrayRef<int64_t> opShape,
                                    TypeRange elementalTypes) {
  // TODO: Tablegen all this.
  Type f16 = b.getF16Type();
  Type f32 = b.getF32Type();
  if (opShape == ArrayRef<int64_t>{16, 8, 4} &&
      elementalTypes == TypeRange{f32, f32, f32}) {
    return MmaSyncInfo{std::make_tuple(&MmaSyncBuilder::m16n8k4tf32Lhs,
                                       &MmaSyncBuilder::m16n8k4tf32Rhs,
                                       &MmaSyncBuilder::m16n8k4tf32Res),
                       makeVectorShapes({2, 1}, {1, 1}, {2, 2}),
                       SmallVector<int64_t>{opShape.begin(), opShape.end()},
                       /*tf32Enabled=*/true};
  }
  // This is the version with f16 accumulation.
  // TODO: version with f32 accumulation.
  if (opShape == ArrayRef<int64_t>{16, 8, 16} &&
      elementalTypes == TypeRange{f16, f16, f16}) {
    return MmaSyncInfo{std::make_tuple(&MmaSyncBuilder::m16n8k16f16Lhs,
                                       &MmaSyncBuilder::m16n8k16f16Rhs,
                                       &MmaSyncBuilder::m16n8k16f16Res),
                       makeVectorShapes({4, 2}, {2, 2}, {2, 2}),
                       SmallVector<int64_t>{opShape.begin(), opShape.end()},
                       /*tf32Enabled=*/false};
  }
  return failure();
}

FailureOr<Operation *> MmaSyncBuilder::buildMmaSync(LinalgOp linalgOp) {
  Value lhsMemref = linalgOp.getDpsInputOperand(0)->get();
  Value rhsMemref = linalgOp.getDpsInputOperand(1)->get();
  Value resMemref = linalgOp.getDpsInitOperand(0)->get();
  assert(lhsMemref.getType().cast<MemRefType>().getRank() == 2 &&
         "expected lhs to be a 2D memref");
  assert(rhsMemref.getType().cast<MemRefType>().getRank() == 2 &&
         "expected rhs to be a 2D memref");
  assert(resMemref.getType().cast<MemRefType>().getRank() == 2 &&
         "expected res to be a 2D memref");

  int64_t m = cast<MemRefType>(lhsMemref.getType()).getShape()[0];
  int64_t n = cast<MemRefType>(rhsMemref.getType()).getShape()[1];
  int64_t k = cast<MemRefType>(lhsMemref.getType()).getShape()[1];
  Type lhsType = getElementTypeOrSelf(lhsMemref.getType());
  Type rhsType = getElementTypeOrSelf(rhsMemref.getType());
  Type resType = getElementTypeOrSelf(resMemref.getType());

  FailureOr<MmaSyncInfo> maybeInfo =
      getIndexCalculators({m, n, k}, {lhsType, rhsType, resType});
  if (failed(maybeInfo))
    return failure();

  MmaSyncInfo info = *maybeInfo;
  auto [lhsIndexFn, rhsIndexFn, resIndexFn] = info.indexFns;
  auto [lhsShape, rhsShape, resShape] = info.vectorShapes;
  Value lhs = buildMmaSyncMemrefLoadOperand(b, loc, laneId, lhsMemref,
                                            lhsIndexFn, lhsShape);
  Value rhs = buildMmaSyncMemrefLoadOperand(b, loc, laneId, rhsMemref,
                                            rhsIndexFn, rhsShape);
  Value res = buildMmaSyncMemrefLoadOperand(b, loc, laneId, resMemref,
                                            resIndexFn, resShape);
  res = b.create<nvgpu::MmaSyncOp>(loc, lhs, rhs, res, info.mmaShape,
                                   info.tf32Enabled);
  buildMmaSyncMemrefStoreOperand(b, loc, res, laneId, resMemref, resIndexFn,
                                 resShape);
  return res.getDefiningOp();
}

DiagnosedSilenceableFailure transform::RewriteMatmulAsMmaSyncOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp linalgOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  bool fail = true;
  // TODO: more robust detection of matmulOp, with transposes etc.
  if (auto matmulOp = isa<linalg::MatmulOp>(linalgOp.getOperation())) {
    Location loc = linalgOp.getLoc();
    // TODO: more robust computation of laneId, for now assume a single warp.
    Value laneId = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), gpu::Dimension::x);
    if (succeeded(MmaSyncBuilder(rewriter, loc, laneId).buildMmaSync(linalgOp)))
      fail = false;
  }

  if (fail) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "unsupported target op: " << linalgOp;
    diag.attachNote(linalgOp->getLoc()) << "target op";
    return diag;
  }

  rewriter.eraseOp(linalgOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class NVGPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          NVGPUTransformDialectExtension> {
public:
  NVGPUTransformDialectExtension() {
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<nvgpu::NVGPUDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp.inc"

void mlir::nvgpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<NVGPUTransformDialectExtension>();
}
