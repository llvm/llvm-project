//===- PrepareVectorContractLoops.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "x86-prepare-vector-contract-loops"

using namespace mlir;

static vector::ContractionOp getCandidate(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1) {
    LDBG() << "reject loop at " << loop.getLoc() << ": expected one iter_arg";
    return {};
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto contract = dyn_cast<vector::ContractionOp>(&op);
    if (!contract)
      continue;
    if (contract.getKind() != vector::CombiningKind::ADD ||
        contract.getAcc() != loop.getRegionIterArgs().front() ||
        !contract.getAcc().hasOneUse() || !contract.getResult().hasOneUse() ||
        yield->getNumOperands() != 1 ||
        yield->getOperand(0) != contract.getResult()) {
      LDBG() << "reject contraction at " << contract.getLoc()
             << ": not a single loop-carried accumulation";
      continue;
    }

    auto iterators = contract.getIteratorTypesArray();
    auto maps = contract.getIndexingMapsArray();
    MLIRContext *ctx = contract.getContext();
    auto m = getAffineDimExpr(0, ctx);
    auto n = getAffineDimExpr(1, ctx);
    auto k = getAffineDimExpr(2, ctx);
    auto par = vector::IteratorType::parallel;
    auto red = vector::IteratorType::reduction;
    if (llvm::equal(iterators, ArrayRef{par, par, red}) &&
        maps[0] == AffineMap::get(3, 0, {m, k}, ctx) &&
        maps[1] == AffineMap::get(3, 0, {k, n}, ctx) &&
        maps[2] == AffineMap::get(3, 0, {m, n}, ctx))
      return contract;

    auto v = getAffineDimExpr(3, ctx);
    if (x86::isInVnniLayout(contract, maps) &&
        llvm::equal(iterators, ArrayRef{par, par, red, red}) &&
        maps[0] == AffineMap::get(4, 0, {m, k, v}, ctx) &&
        maps[1] == AffineMap::get(4, 0, {k, n, v}, ctx) &&
        maps[2] == AffineMap::get(4, 0, {m, n}, ctx))
      return contract;
    LDBG() << "reject contraction at " << contract.getLoc()
           << ": noncanonical matmul layout or iterators";
  }
  return {};
}

static bool isSimpleReadOutsideLoop(vector::TransferReadOp read,
                                    scf::ForOp loop) {
  return read && read->getParentOp() == loop &&
         isa<MemRefType>(read.getBase().getType()) &&
         loop.isDefinedOutsideOfLoop(read.getBase()) &&
         read.getPermutationMap().isMinorIdentity() && !read.getMask() &&
         !read.hasOutOfBoundsDim();
}

namespace {
struct ExpandedRead {
  vector::TransferReadOp read;
  MemRefType type;
  SmallVector<ReassociationIndices> reassociation;
  int64_t lastIndex;
  // Static view of a dynamic last dimension, or null.
  MemRefType view;
};
} // namespace

static std::optional<ExpandedRead>
getExpandableRead(vector::TransferReadOp read, unsigned factor,
                  scf::ForOp loop) {
  if (!isSimpleReadOutsideLoop(read, loop))
    return std::nullopt;
  auto type = cast<MemRefType>(read.getBase().getType());
  if (type.getElementType() != read.getVectorType().getElementType() ||
      type.getRank() < 2)
    return std::nullopt;
  auto lastIndex = getConstantIntValue(read.getIndices().back());
  if (!lastIndex || *lastIndex < 0 || *lastIndex % factor != 0)
    return std::nullopt;

  MemRefType view;
  if (type.isDynamicDim(type.getRank() - 1)) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (read.getVectorType().isScalable() ||
        failed(type.getStridesAndOffset(strides, offset)))
      return std::nullopt;
    // The full dimension has a dynamic size, but this in-bounds read ends at
    // a known index. Create a view whose last dimension ends there and give
    // the divisibility check below a static size to work with.
    SmallVector<int64_t> shape(type.getShape());
    shape.back() = *lastIndex + read.getVectorType().getShape().back();
    view = type = MemRefType::get(
        shape, type.getElementType(),
        StridedLayoutAttr::get(type.getContext(), offset, strides),
        type.getMemorySpace());
  }
  if (type.getShape().back() % factor != 0)
    return std::nullopt;

  SmallVector<int64_t> shape(type.getShape());
  shape.back() /= factor;
  shape.push_back(factor);
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < type.getRank(); ++i)
    reassociation.push_back(i == type.getRank() - 1
                                ? ReassociationIndices{i, i + 1}
                                : ReassociationIndices{i});
  FailureOr<MemRefType> expanded =
      memref::ExpandShapeOp::computeExpandedType(type, shape, reassociation);
  if (failed(expanded))
    return std::nullopt;
  return ExpandedRead{read, *expanded, reassociation, *lastIndex / factor,
                      view};
}

static Value buildExpandedRead(PatternRewriter &rewriter, Location loc,
                               ExpandedRead &info, VectorType resultType) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(info.read->getParentOp());
  Value base = info.read.getBase();
  if (info.view) {
    // Unlike memref.subview, memref.reinterpret_cast is valid even when
    // the loop runs zero times (which makes the view out of bounds).
    auto metadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, base);
    SmallVector<OpFoldResult> sizes = metadata.getConstifiedMixedSizes();
    sizes.back() = rewriter.getIndexAttr(info.view.getShape().back());
    base = memref::ReinterpretCastOp::create(
        rewriter, loc, info.view, base, metadata.getConstifiedMixedOffset(),
        sizes, metadata.getConstifiedMixedStrides());
  }
  Value expanded = memref::ExpandShapeOp::create(rewriter, loc, info.type, base,
                                                 info.reassociation);
  Value lastIndex =
      arith::ConstantIndexOp::create(rewriter, loc, info.lastIndex);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  rewriter.setInsertionPoint(info.read);
  SmallVector<Value> indices(info.read.getIndices());
  indices.back() = lastIndex;
  indices.push_back(zero);
  auto packed = vector::TransferReadOp::create(
      rewriter, loc, resultType, expanded, indices, info.read.getPadding());
  packed.setInBoundsAttr(
      rewriter.getBoolArrayAttr(SmallVector<bool>(resultType.getRank(), true)));
  return packed;
}

namespace {
// Fold RHS unpacking into a VNNI contraction:
//   - Match a packed RHS read transposed and flattened for a flat matmul.
//   - Split the LHS memref's K dimension and reread it in packed lanes.
//     A dynamic K dimension is first narrowed to the read's static extent.
//   - Contract the packed reads directly over K and the VNNI lane dimension.
// For bf16, 32x32 LHS and RHS become 32x16x2 LHS and 16x32x2 RHS.
struct FoldVNNIUnpacking : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contract,
                                PatternRewriter &rewriter) const override {
    auto loop = dyn_cast<scf::ForOp>(contract->getParentOp());
    if (!loop)
      return rewriter.notifyMatchFailure(contract,
                                         "not directly in a for loop");
    if (getCandidate(loop) != contract)
      return rewriter.notifyMatchFailure(
          contract, "not the loop's canonical contraction");

    auto lhsType = contract.getLhsType();
    auto rhsType = contract.getRhsType();
    auto accType = dyn_cast<VectorType>(contract.getAccType());
    Type element = lhsType.getElementType();
    if (!accType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        accType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          contract, "expected 2D operands and accumulator");
    if (lhsType.isScalable() || rhsType.isScalable())
      return rewriter.notifyMatchFailure(contract, "scalable operand vectors");
    if (element != rhsType.getElementType())
      return rewriter.notifyMatchFailure(contract,
                                         "operand element types differ");
    if (!(element.isBF16() || element.isInteger(8)))
      return rewriter.notifyMatchFailure(contract,
                                         "expected bf16 or i8 operands");
    unsigned factor = 32 / element.getIntOrFloatBitWidth();
    if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
      return rewriter.notifyMatchFailure(contract, "K dimensions differ");
    if (lhsType.getDimSize(1) % factor != 0)
      return rewriter.notifyMatchFailure(
          contract, "K dimension is not divisible by VNNI factor");

    auto shapeCast = contract.getRhs().getDefiningOp<vector::ShapeCastOp>();
    auto transpose =
        shapeCast ? shapeCast.getSource().getDefiningOp<vector::TransposeOp>()
                  : vector::TransposeOp();
    if (!transpose || transpose->getParentOp() != loop)
      return rewriter.notifyMatchFailure(
          contract, "RHS is not an in-loop transpose followed by shape_cast");
    VectorType rhsPackedType =
        VectorType::get({lhsType.getDimSize(1) / factor, rhsType.getDimSize(1),
                         int64_t(factor)},
                        element);
    auto sourceType = cast<VectorType>(transpose.getVector().getType());
    if (sourceType.getRank() < rhsPackedType.getRank())
      return rewriter.notifyMatchFailure(contract,
                                         "RHS source rank is too small");
    unsigned leading = sourceType.getRank() - rhsPackedType.getRank();
    if (!llvm::all_of(sourceType.getShape().take_front(leading),
                      [](int64_t dim) { return dim == 1; }) ||
        !llvm::equal(sourceType.getShape().drop_front(leading),
                     rhsPackedType.getShape()))
      return rewriter.notifyMatchFailure(contract,
                                         "RHS source is not VNNI packed");
    for (unsigned i = 0; i < leading + 1; ++i)
      if (transpose.getPermutation()[i] != i)
        return rewriter.notifyMatchFailure(
            contract, "transpose reorders leading or K dimensions");
    if (transpose.getPermutation()[leading + 1] != leading + 2 ||
        transpose.getPermutation()[leading + 2] != leading + 1)
      return rewriter.notifyMatchFailure(
          contract, "transpose does not swap N and VNNI lanes");

    auto lhsRead = contract.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto lhs = getExpandableRead(lhsRead, factor, loop);
    auto broadcast = transpose.getVector().getDefiningOp<vector::BroadcastOp>();
    Value packedSource =
        broadcast ? broadcast.getSource() : transpose.getVector();
    if (broadcast && packedSource.getType() != rhsPackedType)
      return rewriter.notifyMatchFailure(contract,
                                         "broadcast source is not VNNI packed");
    auto packedCast = packedSource.getDefiningOp<vector::ShapeCastOp>();
    auto rhsRead =
        packedCast
            ? packedCast.getSource().getDefiningOp<vector::TransferReadOp>()
            : packedSource.getDefiningOp<vector::TransferReadOp>();
    if (!lhs)
      return rewriter.notifyMatchFailure(
          contract, "LHS read cannot be expanded into VNNI layout");
    if (!isSimpleReadOutsideLoop(rhsRead, loop))
      return rewriter.notifyMatchFailure(contract,
                                         "RHS is not a simple transfer read");
    std::optional<ExpandedRead> splitRhs;
    if (packedCast) {
      auto flatType = rhsRead.getVectorType();
      if (flatType.getRank() != 2 ||
          flatType.getDimSize(0) != rhsPackedType.getDimSize(0) ||
          flatType.getDimSize(1) % factor != 0 ||
          flatType.getDimSize(1) / factor != rhsPackedType.getDimSize(1))
        return rewriter.notifyMatchFailure(
            contract, "flat RHS shape is not VNNI compatible");
      if (!(splitRhs = getExpandableRead(rhsRead, factor, loop)))
        return rewriter.notifyMatchFailure(
            contract, "RHS read cannot be expanded into VNNI layout");
    } else {
      if (rhsRead.getVectorType() != rhsPackedType)
        return rewriter.notifyMatchFailure(contract,
                                           "RHS read is not VNNI packed");
      if (getConstantIntValue(rhsRead.getIndices().back()) != 0)
        return rewriter.notifyMatchFailure(
            contract, "packed RHS has nonzero inner offset");
    }

    Location loc = contract.getLoc();
    VectorType lhsPackedType =
        VectorType::get({lhsType.getDimSize(0), lhsType.getDimSize(1) / factor,
                         int64_t(factor)},
                        element);
    Value packedLhs = buildExpandedRead(rewriter, loc, *lhs, lhsPackedType);
    Value packedRhs =
        splitRhs ? buildExpandedRead(rewriter, loc, *splitRhs, rhsPackedType)
                 : rhsRead.getResult();
    rewriter.setInsertionPoint(contract);
    auto ctx = rewriter.getContext();
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    auto v = rewriter.getAffineDimExpr(3);
    auto vnniMaps =
        rewriter.getAffineMapArrayAttr({AffineMap::get(4, 0, {m, k, v}, ctx),
                                        AffineMap::get(4, 0, {k, n, v}, ctx),
                                        AffineMap::get(4, 0, {m, n}, ctx)});
    auto iterators = rewriter.getArrayAttr(
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contract, packedLhs, packedRhs, contract.getAcc(), vnniMaps, iterators,
        contract.getKind(), contract.getFastmath());
    return success();
  }
};
} // namespace

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldVNNIUnpacking>(patterns.getContext());
}
