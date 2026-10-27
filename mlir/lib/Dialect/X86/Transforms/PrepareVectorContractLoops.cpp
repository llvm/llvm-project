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
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVectorExtras.h"
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
    LDBG() << "reject contraction at " << contract.getLoc()
           << ": noncanonical matmul layout or iterators";
  }
  return {};
}

// Look through reshapes/broadcasts without changing the row-major element
// order. A broadcast is safe to skip if it adds unit dimensions only.
static Value skipReshapes(Value v) {
  while (isa_and_present<vector::ShapeCastOp, vector::BroadcastOp>(
      v.getDefiningOp())) {
    Value source = v.getDefiningOp()->getOperand(0);
    auto sourceType = dyn_cast<VectorType>(source.getType());
    if (!sourceType || sourceType.getNumElements() !=
                           cast<VectorType>(v.getType()).getNumElements())
      break;
    v = source;
  }
  return v;
}

// Find the source and parity of an even/odd split: v[i] = src[2 * i + j].
// Match either pattern below (unit dimensions may also be present):
//
//   %even, %odd = vector.deinterleave %src
//       : vector<4x8xbf16> -> vector<4x4xbf16>
//
// or:
//
//   %packed = vector.shape_cast %src
//       : vector<4x8xbf16> to vector<4x4x2xbf16>
//   %split = vector.transpose %packed, [2, 0, 1]
//       : vector<4x4x2xbf16> to vector<2x4x4xbf16>
//   %even = vector.extract %split[0]
//       : vector<4x4xbf16> from vector<2x4x4xbf16>
//   %odd = vector.extract %split[1]
//       : vector<4x4xbf16> from vector<2x4x4xbf16>
static std::optional<std::pair<Value, int64_t>> matchEvenOrOdd(Value v) {
  v = skipReshapes(v);
  if (auto deinterleave = v.getDefiningOp<vector::DeinterleaveOp>())
    return std::pair(skipReshapes(deinterleave.getSource()),
                     int64_t(cast<OpResult>(v).getResultNumber()));

  auto extract = v.getDefiningOp<vector::ExtractOp>();
  auto transpose =
      extract ? extract.getSource().getDefiningOp<vector::TransposeOp>()
              : vector::TransposeOp();
  auto type = dyn_cast<VectorType>(v.getType());
  if (!transpose || !type || extract.hasDynamicPosition() ||
      transpose.getSourceVectorType().isScalable())
    return std::nullopt;

  ArrayRef<int64_t> shape = transpose.getSourceVectorType().getShape();
  auto isNonUnit = [&](int64_t dim) { return shape[dim] != 1; };
  SmallVector<int64_t> order =
      llvm::filter_to_vector(llvm::seq<int64_t>(0, shape.size()), isNonUnit);
  if (order.empty() || shape[order.back()] != 2)
    return std::nullopt;

  std::rotate(order.begin(), std::prev(order.end()), order.end());
  ArrayRef<int64_t> perm = transpose.getPermutation();
  if (llvm::filter_to_vector(perm, isNonUnit) != order)
    return std::nullopt;

  // The dimension of size 2 is now first among the non-unit dimensions.
  // Keeping exactly half the elements means the extract indexes that
  // dimension; any other dimensions it indexes must be unit dimensions.
  if (2 * type.getNumElements() !=
      transpose.getSourceVectorType().getNumElements())
    return std::nullopt;

  int64_t j =
      extract
          .getStaticPosition()[llvm::find(perm, order.front()) - perm.begin()];
  if (j != 0 && j != 1)
    return std::nullopt;

  return std::pair(skipReshapes(transpose.getVector()), j);
}

// Find the even and odd inputs of either pattern below. The slice inserts may
// appear in either order.
//
//   %joined = vector.interleave %even, %odd
//       : vector<4x1xbf16> -> vector<4x2xbf16>
//
// or:
//
//   %poison = ub.poison : vector<4x2xbf16>
//   %partial = vector.insert_strided_slice %odd, %poison
//       offsets = [0, 1], strides = [1, 1]
//       : vector<4x1xbf16> into vector<4x2xbf16>
//   %joined = vector.insert_strided_slice %even, %partial
//       offsets = [0, 0], strides = [1, 1]
//       : vector<4x1xbf16> into vector<4x2xbf16>
static std::optional<std::array<Value, 2>> matchInterleave(Value v) {
  v = skipReshapes(v);
  if (auto interleave = v.getDefiningOp<vector::InterleaveOp>())
    return std::array<Value, 2>{interleave.getLhs(), interleave.getRhs()};

  // Both slices must be complete and disjoint so the original destination
  // contributes no elements to the result.
  std::array<Value, 2> halves;
  for (int i = 0; i < 2; ++i) {
    auto insert = v.getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insert)
      return std::nullopt;

    ArrayRef<int64_t> shape = insert.getDestVectorType().getShape();
    SmallVector<int64_t> slice(shape.drop_back());
    slice.push_back(1);
    auto offsets = extractFromIntegerArrayAttr<int64_t>(insert.getOffsets());
    int64_t j = offsets.back();
    if (shape.back() != 2 ||
        insert.getSourceVectorType().getShape() != ArrayRef(slice) || halves[j])
      return std::nullopt;

    halves[j] = insert.getValueToStore();
    v = insert.getDest();
  }
  return halves;
}

// Find the packed RHS so the contraction can use it without unpacking.
// For a K x N RHS and VNNI factor f, require
//   rhs[f * r + j][c] = src[r][c][j], with src viewed as K/f x N x f.
// Match transpose/shape_cast unpacking (shown for f = 2):
//
//   %packed = vector.shape_cast %src
//       : vector<4x6xbf16> to vector<4x3x2xbf16>
//   %transposed = vector.transpose %packed, [0, 2, 1]
//       : vector<4x3x2xbf16> to vector<4x2x3xbf16>
//   %rhs = vector.shape_cast %transposed
//       : vector<4x2x3xbf16> to vector<8x3xbf16>
//
// or, for f = 2, split/transpose/join/transpose unpacking:
//
//   %even, %odd = vector.deinterleave %src
//       : vector<4x6xbf16> -> vector<4x3xbf16>
//   %even_t = vector.transpose %even, [1, 0]
//       : vector<4x3xbf16> to vector<3x4xbf16>
//   %odd_t = vector.transpose %odd, [1, 0]
//       : vector<4x3xbf16> to vector<3x4xbf16>
//   %joined = vector.interleave %even_t, %odd_t
//       : vector<3x4xbf16> -> vector<3x8xbf16>
//   %rhs = vector.transpose %joined, [1, 0]
//       : vector<3x8xbf16> to vector<8x3xbf16>
static Value matchUnpackedRhs(Value rhs, int64_t factor) {
  auto type = cast<VectorType>(rhs.getType());
  int64_t k = type.getDimSize(0) / factor, n = type.getDimSize(1);
  auto transpose = skipReshapes(rhs).getDefiningOp<vector::TransposeOp>();
  if (!transpose)
    return {};

  ArrayRef<int64_t> shape = transpose.getSourceVectorType().getShape();
  ArrayRef<int64_t> perm = transpose.getPermutation();
  if (shape.size() >= 3) {
    size_t leading = shape.size() - 3;
    auto expected = llvm::to_vector(llvm::seq<int64_t>(0, shape.size()));
    std::swap(expected[leading + 1], expected[leading + 2]);
    bool isUnpacking =
        llvm::all_of(shape.take_front(leading),
                     [](int64_t dim) { return dim == 1; }) &&
        shape.drop_front(leading) == ArrayRef<int64_t>{k, n, factor} &&
        perm == ArrayRef(expected);
    return isUnpacking ? skipReshapes(transpose.getVector()) : Value();
  }

  const int64_t swap[] = {1, 0};
  if (factor != 2 || perm != ArrayRef(swap) ||
      shape != ArrayRef<int64_t>{n, 2 * k})
    return {};

  std::optional<std::array<Value, 2>> halves =
      matchInterleave(transpose.getVector());
  if (!halves)
    return {};

  Value src;
  for (auto [j, half] : llvm::enumerate(*halves)) {
    auto transpose = skipReshapes(half).getDefiningOp<vector::TransposeOp>();
    if (!transpose || transpose.getPermutation() != ArrayRef(swap) ||
        transpose.getSourceVectorType().getShape() != ArrayRef<int64_t>{k, n})
      return {};

    auto evenOrOdd = matchEvenOrOdd(transpose.getVector());
    if (!evenOrOdd || evenOrOdd->second != int64_t(j) ||
        (src && evenOrOdd->first != src))
      return {};

    src = evenOrOdd->first;
  }
  return src;
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
//   - Match a packed RHS read unpacked for a flat matmul.
//   - Split the LHS memref's K dimension and reread it in VNNI layout.
//     A dynamic K dimension is first narrowed to the read's static extent.
//   - Contract the packed reads directly over K and the VNNI dimension.
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
    Type element = lhsType.getElementType();
    if (lhsType.isScalable() || rhsType.isScalable())
      return rewriter.notifyMatchFailure(contract, "scalable operand vectors");
    if (!(element.isBF16() || element.isInteger(8)))
      return rewriter.notifyMatchFailure(contract,
                                         "expected bf16 or i8 operands");
    unsigned factor = 32 / element.getIntOrFloatBitWidth();
    if (lhsType.getDimSize(1) % factor != 0)
      return rewriter.notifyMatchFailure(
          contract, "K dimension is not divisible by VNNI factor");

    Value packedSource = matchUnpackedRhs(contract.getRhs(), factor);
    if (!packedSource)
      return rewriter.notifyMatchFailure(contract,
                                         "RHS does not unpack a packed vector");
    VectorType rhsPackedType =
        VectorType::get({lhsType.getDimSize(1) / factor, rhsType.getDimSize(1),
                         int64_t(factor)},
                        element);
    auto lhsRead = contract.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto lhs = getExpandableRead(lhsRead, factor, loop);
    auto rhsRead = packedSource.getDefiningOp<vector::TransferReadOp>();
    if (!lhs)
      return rewriter.notifyMatchFailure(
          contract, "LHS read cannot be expanded into VNNI layout");
    if (!isSimpleReadOutsideLoop(rhsRead, loop))
      return rewriter.notifyMatchFailure(contract,
                                         "RHS is not a simple transfer read");
    std::optional<ExpandedRead> splitRhs;
    if (rhsRead.getVectorType() != rhsPackedType) {
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
    LDBG() << "Replaced contract with VNNI-compatible contraction";
    return success();
  }
};
} // namespace

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldVNNIUnpacking>(patterns.getContext());
}
