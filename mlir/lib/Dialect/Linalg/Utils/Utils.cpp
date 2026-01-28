//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-utils"

using namespace mlir;
using namespace presburger;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::scf;

namespace {

// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<OpFoldResult> tileSizes) : tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZeroInteger(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(cast<AffineConstantExpr>(expr.getRHS()).getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled = false;
  ArrayRef<OpFoldResult> tileSizes;
};

} // namespace

static bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

std::optional<RegionMatcher::BinaryOpKind>
RegionMatcher::matchAsScalarBinaryOp(GenericOp op) {
  auto &region = op.getRegion();
  if (!region.hasOneBlock())
    return std::nullopt;

  Block &block = region.front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isSignlessIntOrFloat() ||
      !block.getArgument(1).getType().isSignlessIntOrFloat())
    return std::nullopt;

  auto &ops = block.getOperations();
  if (!llvm::hasSingleElement(block.without_terminator()))
    return std::nullopt;

  using mlir::matchers::m_Val;
  auto a = m_Val(block.getArgument(0));
  auto b = m_Val(block.getArgument(1));

  auto addPattern = m_Op<linalg::YieldOp>(m_Op<arith::AddIOp>(a, b));
  if (addPattern.match(&ops.back()))
    return BinaryOpKind::IAdd;

  return std::nullopt;
}

/// Explicit instantiation of loop nest generator for different loop types.
template struct mlir::linalg::GenerateLoopNest<scf::ForOp>;
template struct mlir::linalg::GenerateLoopNest<scf::ParallelOp>;
template struct mlir::linalg::GenerateLoopNest<AffineForOp>;

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(OpBuilder &builder, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
}

//===----------------------------------------------------------------------===//
// General utilities
//===----------------------------------------------------------------------===//
//
/// The permutation can be obtained from two permutations:
///   a) Compute the permutation vector to move the last `numPackedDims` into
///      the `innerPosDims` of a shape of rank `rank`.
///   b) Compute the permutation vector to move outer dims if the
///      `outerPerm` parameter is not empty.
/// Apply (b) permutation on (a) permutation to get the final permutation.
static SmallVector<int64_t>
computePackUnPackPerm(int64_t rank, ArrayRef<int64_t> &innerDimsPos,
                      ArrayRef<int64_t> &outerPerm,
                      PackingMetadata &packingMetadata) {
  int64_t numPackedDims = innerDimsPos.size();
  auto lastDims =
      llvm::to_vector(llvm::seq<int64_t>(rank - numPackedDims, rank));
  packingMetadata = computePackingMetadata(rank, innerDimsPos);
  SmallVector<int64_t> innerPositionsPerm =
      computePermutationVector(rank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerPerm.empty())
    applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm =
      computePermutationVector(rank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packInverseDestPermutation = innerPositionsPerm;
  applyPermutationToVector(packInverseDestPermutation, outerPositionPerm);
  return packInverseDestPermutation;
}

namespace mlir {
namespace linalg {

SmallVector<int64_t> getPackInverseDestPerm(PackOp packOp,
                                            PackingMetadata &metadata) {

  int64_t packedRank = packOp.getDestType().getRank();
  ArrayRef<int64_t> innerDimPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = packOp.getOuterDimsPerm();
  SmallVector<int64_t> packInvDestPerm =
      computePackUnPackPerm(packedRank, innerDimPos, outerPerm, metadata);
  return packInvDestPerm;
}

SmallVector<int64_t> getUnPackInverseSrcPerm(UnPackOp unpackOp,
                                             PackingMetadata &metadata) {
  int64_t packedRank = unpackOp.getSourceType().getRank();
  ArrayRef<int64_t> innerDimPos = unpackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = unpackOp.getOuterDimsPerm();
  SmallVector<int64_t> unpackInvSrcPerm =
      computePackUnPackPerm(packedRank, innerDimPos, outerPerm, metadata);
  return unpackInvSrcPerm;
}

bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!r.hasOneBlock())
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<arith::ConstantOp, func::ConstantOp, tensor::ExtractOp,
              linalg::YieldOp, linalg::IndexOp, AffineApplyOp>(op) ||
          OpTrait::hasElementwiseMappableTraits(&op)) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

bool isElementwise(LinalgOp op) {
  if (op.getNumLoops() != op.getNumParallelLoops())
    return false;

  if (!allIndexingsAreProjectedPermutation(op))
    return false;

  // TODO: relax the restrictions on indexing map.
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    if (!op.getMatchingIndexingMap(&opOperand).isPermutation())
      return false;
  }
  return hasOnlyScalarElementwiseOp(op->getRegion(0));
}

bool isParallelIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::parallel;
}

bool isReductionIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::reduction;
}

//===----------------------------------------------------------------------===//
// Convolution matcher utilities
//===----------------------------------------------------------------------===//

/// Returns the BlockArgument that leads to `val`, if any. Traverses optional
/// ext*/sitofp ops.
static BlockArgument getBlockArgumentWithOptionalCastOps(Value val) {
  BlockArgument blockArg = dyn_cast<BlockArgument>(val);
  if ((blockArg))
    return blockArg;

  Operation *defOp = val.getDefiningOp();
  if (!dyn_cast_if_present<arith::ExtFOp>(defOp) &&
      !dyn_cast_if_present<arith::ExtSIOp>(defOp) &&
      !dyn_cast_if_present<arith::ExtUIOp>(defOp) &&
      !dyn_cast_if_present<arith::SIToFPOp>(defOp)) {
    return nullptr;
  }
  return dyn_cast<BlockArgument>(defOp->getOperand(0));
}

/// Utility function to match the zero point offset body of quantized
/// convolution ops.
///
/// Quantized convolutions have a body of the form:
///   %out + ((%input - %inputZp) * (%filter - %filterZp))
/// where:
///   - %input is the input tensor element (block arg 0)
///   - %filter is the filter tensor element (block arg 1)
///   - %inputZp is the input zero-point scalar (block arg 2)
///   - %filterZp is the filter zero-point scalar (block arg 3)
///   - %out is the output accumulator (block arg 4)
///
/// This function verifies that the multiplication operands are subtraction
/// operations matching this pattern.
static bool bodyMatcherForZeroPointOffsets(Operation *addOp, Operation *mulOp,
                                           Block *body) {
  // The multiplication should have two subtraction operands:
  // one for (input - inputZp) and one for (filter - filterZp).
  Operation *inputSubOp = mulOp->getOperand(0).getDefiningOp();
  if (!isa_and_present<arith::SubIOp, arith::SubFOp>(inputSubOp))
    return false;

  Operation *filterSubOp = mulOp->getOperand(1).getDefiningOp();
  if (!isa_and_present<arith::SubIOp, arith::SubFOp>(filterSubOp))
    return false;

  // Extract block arguments from subtraction operands.
  BlockArgument inputBlockArg =
      getBlockArgumentWithOptionalCastOps(inputSubOp->getOperand(0));
  BlockArgument inputZpBlockArg =
      getBlockArgumentWithOptionalCastOps(inputSubOp->getOperand(1));
  BlockArgument filterBlockArg =
      getBlockArgumentWithOptionalCastOps(filterSubOp->getOperand(0));
  BlockArgument filterZpBlockArg =
      getBlockArgumentWithOptionalCastOps(filterSubOp->getOperand(1));
  BlockArgument outBlockArg =
      getBlockArgumentWithOptionalCastOps(addOp->getOperand(0));

  // Verify all block arguments are valid.
  if (!inputBlockArg || !inputZpBlockArg || !filterBlockArg ||
      !filterZpBlockArg || !outBlockArg)
    return false;

  // Verify all block arguments belong to the convolution body.
  if (inputBlockArg.getOwner() != body || inputZpBlockArg.getOwner() != body ||
      filterBlockArg.getOwner() != body ||
      filterZpBlockArg.getOwner() != body || outBlockArg.getOwner() != body)
    return false;

  // Verify block arguments have expected indices:
  // arg0: input, arg1: filter, arg2: inputZp, arg3: filterZp, arg4: output
  if (inputBlockArg.getArgNumber() != 0 || filterBlockArg.getArgNumber() != 1 ||
      inputZpBlockArg.getArgNumber() != 2 ||
      filterZpBlockArg.getArgNumber() != 3 || outBlockArg.getArgNumber() != 4)
    return false;

  return true;
}

/// Utility to match block body for convolution ops.
/// The body is thus expected to yield :-
///     %out + (%lhs * %rhs)
///   where: %lhs, %rhs and %out are block arguments and
///          %lhs and %rhs can have optional upcast operation.
/// For i1 element types, the pattern matches:
///     %out | (%lhs & %rhs)
///   using arith.ori for accumulation and arith.andi for multiplication.
/// NOTE: In case of zero point offset convolution ops %lhs and %rhs would be :-
///       %input - %input_scalar
///          where, %input_scalar can have optional upcast operation.
static bool bodyMatcherForConvolutionOps(Value yieldVal, Block *body,
                                         bool containsZeroPointOffset = false) {
  bool isOrOp = false;
  Operation *accOp = yieldVal.getDefiningOp();
  if (!isa_and_present<arith::AddIOp, arith::AddFOp>(accOp)) {
    if (!isa_and_present<arith::OrIOp>(accOp))
      return false;
    isOrOp = true;
  }

  Operation *mulOp = accOp->getOperand(1).getDefiningOp();
  if (!isOrOp && !isa_and_present<arith::MulIOp, arith::MulFOp>(mulOp))
    return false;
  if (isOrOp && !isa_and_present<arith::AndIOp>(mulOp))
    return false;

  if (containsZeroPointOffset) {
    return bodyMatcherForZeroPointOffsets(accOp, mulOp, body);
  }
  BlockArgument lhsBlockArg =
      getBlockArgumentWithOptionalCastOps(mulOp->getOperand(0));
  BlockArgument rhsBlockArg =
      getBlockArgumentWithOptionalCastOps(mulOp->getOperand(1));
  BlockArgument outBlockArg =
      getBlockArgumentWithOptionalCastOps(accOp->getOperand(0));
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2)
    return false;
  return true;
}

/// Utility to match block body for linalg.pool* ops.
template <typename... OpTypes>
static bool bodyMatcherForPoolOps(Value yieldVal, Block *body) {
  Operation *defOp = yieldVal.getDefiningOp();
  if (!(isa_and_present<OpTypes>(defOp) || ...))
    return false;

  BlockArgument lhsArg =
      getBlockArgumentWithOptionalCastOps(defOp->getOperand(0));
  BlockArgument rhsArg =
      getBlockArgumentWithOptionalCastOps(defOp->getOperand(1));
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != body ||
      rhsArg.getOwner() != body || lhsArg.getArgNumber() != 2 ||
      rhsArg.getArgNumber() != 0)
    return false;
  return true;
}

static bool bodyMatcherForMaxSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaximumFOp, arith::MaxSIOp>(yieldVal,
                                                                  body);
}

static bool bodyMatcherForMaxUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaxUIOp>(yieldVal, body);
}

static bool bodyMatcherForMinSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinimumFOp, arith::MinSIOp>(yieldVal,
                                                                  body);
}

static bool bodyMatcherForMinUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinUIOp>(yieldVal, body);
}

/// Matches sum pooling body pattern. For i1 element types, arith.ori is used
/// instead of arith.addi/arith.addf for accumulation.
static bool bodyMatcherForSumPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::AddIOp, arith::AddFOp, arith::OrIOp>(
      yieldVal, body);
}

static AffineExpr getAffineMapDim(ArrayAttr indexingMaps, uint32_t mapIndex,
                                  uint32_t dimIndex) {
  auto affineMap = cast<AffineMapAttr>(indexingMaps[mapIndex]).getValue();
  if (dimIndex < affineMap.getNumResults())
    return affineMap.getResult(dimIndex);
  return nullptr;
}

/// Check if `expr` is either:
/// - a dimension expr alone (implying multiplication by 1), or
/// - a multiplication of dimension expr by any positive constant != 1
/// In both cases we will capture the dimension expression into `dim` and
/// return the constant multiplier. Returns -1 in case of a match failure.
static int64_t isDimTimesConstantOrDimOnly(AffineExpr expr, AffineExpr &dim) {
  if ((dim = dyn_cast<AffineDimExpr>(expr)))
    return 1;

  auto mulExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return -1;

  AffineExpr lhs = mulExpr.getLHS();
  AffineExpr rhs = mulExpr.getRHS();

  AffineConstantExpr cst = nullptr;
  if (((dim = dyn_cast<AffineDimExpr>(lhs)) &&
       (cst = dyn_cast<AffineConstantExpr>(rhs))) ||
      ((dim = dyn_cast<AffineDimExpr>(rhs)) &&
       (cst = dyn_cast<AffineConstantExpr>(lhs))))
    return cst.getValue();
  return -1;
}

/// Given an array of AffineMaps `indexingMaps` verify the following
/// commutatively:-
///   indexingMaps[0].getResult(iDim) ==
///         indexingMaps[1].getResult(fDim) * <c0> +
///         indexingMaps[n-1].getResult(oDim) * <c1>
///  where,
///       - c0 and c1 can be any constant,
///       - n is the size of the indexingMaps' array,
///       - 0, 1 and n-1 are input, filter and output map indices respectively,
///       - iDim, fDim and oDim are the input, filter and output dimension
///         indices in their respective indexing maps
///  Example:
///   #inputMap = affine_map<(d0, d1, d2, d3, d4, d5, d6)
///                     -> (d0, d1 * 2 + d4 * 3, d2 + d5, d6)>
///   #filterMap = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
///   #outputMap = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
///
///   Here,
///     #inputMap[1] = #outputMap[1] * 2 + #filterMap[0] * 3
///   Therefore,
///     matchConvDimAddExprPattern(indexingMaps, 1, 0, 1, dilation, stride)
///     would return true and update dilation = 3 and stride = 2
static bool matchConvDimAddExprPattern(ArrayAttr indexingMaps, unsigned iDim,
                                       unsigned fDim, unsigned oDim,
                                       int64_t &dilation, int64_t &stride) {
  unsigned inputMapIdx = 0, filterMapIdx = 1,
           outputMapIdx = indexingMaps.size() - 1;
  AffineExpr inpExpr = getAffineMapDim(indexingMaps, inputMapIdx, iDim);
  auto addExpr = dyn_cast_or_null<AffineBinaryOpExpr>(inpExpr);
  if (!addExpr || addExpr.getKind() != AffineExprKind::Add)
    return false;

  AffineExpr dim0, dim1;
  int64_t c0 = isDimTimesConstantOrDimOnly(addExpr.getLHS(), dim0);
  int64_t c1 = isDimTimesConstantOrDimOnly(addExpr.getRHS(), dim1);

  if (c0 == -1 || c1 == -1)
    return false;
  // Pattern matched with dims and constants extracted.
  AffineExpr fExpr = getAffineMapDim(indexingMaps, filterMapIdx, fDim);
  AffineExpr oExpr = getAffineMapDim(indexingMaps, outputMapIdx, oDim);
  if (dim0 == fExpr && dim1 == oExpr) {
    dilation = c0;
    stride = c1;
    return true;
  }
  if (dim1 == fExpr && dim0 == oExpr) {
    dilation = c1;
    stride = c0;
    return true;
  }
  return false;
}

/// Returns true if the given indexing maps matches with the expected indexing
/// maps.
static bool convLayoutMatches(ArrayRef<ArrayRef<AffineExpr>> mapListExpected,
                              ArrayAttr indexingMaps, MLIRContext *context) {
  SmallVector<AffineMap, 4> expectedIndexingMaps =
      AffineMap::inferFromExprList(mapListExpected, context);
  return indexingMaps ==
         ArrayAttr::get(
             context, llvm::to_vector<4>(llvm::map_range(
                          expectedIndexingMaps, [&](AffineMap m) -> Attribute {
                            return AffineMapAttr::get(m);
                          })));
}

/// Enum representing pooling operation types used by ConvMatcherBuilder.
enum class PoolingType {
  None,
  MaxSigned,
  MaxUnsigned,
  MinSigned,
  MinUnsigned,
  Sum
};

/// Helper class for building convolution op matchers with minimal boilerplate.
/// Reduces repetitive code across Conv1D/2D/3D and Depthwise variants as well
/// as Pooling ops.
///
/// Usage: Create an instance with the op, spatial rank, and output pointers for
/// extracted dilations/strides. Then chain matchStride() calls for each spatial
/// dimension, followed by matchMaps() to verify indexing maps, and finally
/// matchBody() to verify the operation body pattern.
///
/// The `matched` flag starts as `true` and is set to `false` if any match step
/// fails. This allows chaining multiple match calls; once any match fails, all
/// subsequent calls become no-ops and the final result is `false`.
///
/// The `dilations` and `strides` pointers are output parameters that get
/// populated with the extracted dilation and stride values from the operation's
/// indexing maps during matchStride() calls. These values are initially set to
/// 1 for each spatial dimension and updated as patterns are matched.
class ConvMatcherBuilder {
  LinalgOp op;
  MLIRContext *ctx;
  SmallVector<int64_t> *dilations, *strides;
  ArrayAttr indexingMaps;
  PoolingType poolingType;
  bool matched = true;

public:
  ConvMatcherBuilder(LinalgOp op, unsigned spatialRank, SmallVector<int64_t> *d,
                     SmallVector<int64_t> *s,
                     PoolingType poolingType = PoolingType::None)
      : op(op), ctx(op->getContext()), dilations(d), strides(s),
        indexingMaps(op.getIndexingMaps()), poolingType(poolingType) {
    *dilations = SmallVector<int64_t>(spatialRank, 1);
    *strides = SmallVector<int64_t>(spatialRank, 1);
  }

  /// Get affine dimension expression for dimension `i`.
  AffineExpr dim(unsigned i) { return getAffineDimExpr(i, ctx); }

  /// Build strided expression: base * stride[idx] + kernel * dilation[idx].
  AffineExpr strided(AffineExpr base, AffineExpr kernel, unsigned idx) {
    return base * (*strides)[idx] + kernel * (*dilations)[idx];
  }

  /// Match stride/dilation pattern for a spatial dimension.
  /// Returns *this for method chaining.
  ConvMatcherBuilder &matchStride(unsigned iDim, unsigned fDim, unsigned oDim,
                                  unsigned idx) {
    if (matched) {
      matched &= matchConvDimAddExprPattern(indexingMaps, iDim, fDim, oDim,
                                            (*dilations)[idx], (*strides)[idx]);
    }
    return *this;
  }

  /// Match expected indexing maps layout. Returns *this for method chaining.
  ConvMatcherBuilder &matchMaps(ArrayRef<ArrayRef<AffineExpr>> maps) {
    if (matched)
      matched &= convLayoutMatches(maps, indexingMaps, ctx);
    return *this;
  }

  /// Match body pattern. This should be called last.
  bool matchBody(bool containsZeroPointOffset = false) {
    if (!matched)
      return false;
    Block *body = op.getBlock();
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
    switch (poolingType) {
    case PoolingType::None:
      return bodyMatcherForConvolutionOps(yieldOp.getOperand(0), body,
                                          containsZeroPointOffset);
    case PoolingType::MaxSigned:
      return bodyMatcherForMaxSignedPoolOps(yieldOp.getOperand(0), body);
    case PoolingType::MaxUnsigned:
      return bodyMatcherForMaxUnsignedPoolOps(yieldOp.getOperand(0), body);
    case PoolingType::MinSigned:
      return bodyMatcherForMinSignedPoolOps(yieldOp.getOperand(0), body);
    case PoolingType::MinUnsigned:
      return bodyMatcherForMinUnsignedPoolOps(yieldOp.getOperand(0), body);
    case PoolingType::Sum:
      return bodyMatcherForSumPoolOps(yieldOp.getOperand(0), body);
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Matchers for specific convolution operation.
//===----------------------------------------------------------------------===//

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv1DOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (isa<linalg::Conv1DOp>(op)) {
    // Conv1DOp has no strides/dilations attributes, default to 1.
    result.dilations = SmallVector<int64_t>(1, 1);
    result.strides = SmallVector<int64_t>(1, 1);
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr W = m.dim(0);
  AffineExpr w = m.dim(1);

  if (m.matchStride(/*iDim=*/0, /*fDim=*/0, /*oDim=*/0, /*idx=*/0)
          .matchMaps({/*inputMap=*/{m.strided(W, w, 0)},
                      /*filterMap=*/{w},
                      /*outputMap=*/{W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv1DNwcWcfOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv1DNwcWcfOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr F = m.dim(2);
  AffineExpr w = m.dim(3);
  AffineExpr c = m.dim(4);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), c},
                      /*filterMap=*/{w, c, F},
                      /*outputMap=*/{N, W, F}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv1DNcwFcwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv1DNcwFcwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr F = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr c = m.dim(3);
  AffineExpr w = m.dim(4);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, c, m.strided(W, w, 0)},
                      /*filterMap=*/{F, c, w},
                      /*outputMap=*/{N, F, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (isa<linalg::Conv2DOp>(op)) {
    // Conv2DOp has no strides/dilations attributes, default to 1.
    result.dilations = SmallVector<int64_t>(2, 1);
    result.strides = SmallVector<int64_t>(2, 1);
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr H = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr h = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/0, /*fDim=*/0, /*oDim=*/0, /*idx=*/0)
          .matchStride(/*iDim=*/1, /*fDim=*/1, /*oDim=*/1, /*idx=*/1)
          .matchMaps({/*inputMap=*/{m.strided(H, h, 0), m.strided(W, w, 1)},
                      /*filterMap=*/{h, w},
                      /*outputMap=*/{H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwcHwcfOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr F = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);
  AffineExpr c = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), c},
               /*filterMap=*/{h, w, c, F},
               /*outputMap=*/{N, H, W, F}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwcHwcfQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr F = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);
  AffineExpr c = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), c},
               /*filterMap=*/{h, w, c, F},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, H, W, F}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwcFhwcOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwcFhwcOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr F = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);
  AffineExpr c = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/1, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), c},
               /*filterMap=*/{F, h, w, c},
               /*outputMap=*/{N, H, W, F}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwcFhwcQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwcFhwcQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr F = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);
  AffineExpr c = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/1, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), c},
               /*filterMap=*/{F, h, w, c},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, H, W, F}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNchwFchwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr F = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr c = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, c, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{F, c, h, w},
               /*outputMap=*/{N, F, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNchwFchwQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNchwFchwQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr F = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr c = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, c, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{F, c, h, w},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, F, H, W}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNgchwFgchwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNgchwFgchwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr G = m.dim(1);
  AffineExpr F = m.dim(2);
  AffineExpr H = m.dim(3);
  AffineExpr W = m.dim(4);
  AffineExpr c = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/0)
          .matchStride(/*iDim=*/4, /*fDim=*/4, /*oDim=*/4, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, G, c, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{F, G, c, h, w},
               /*outputMap=*/{N, G, F, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNgchwGfchwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNgchwGfchwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr G = m.dim(1);
  AffineExpr F = m.dim(2);
  AffineExpr H = m.dim(3);
  AffineExpr W = m.dim(4);
  AffineExpr c = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/0)
          .matchStride(/*iDim=*/4, /*fDim=*/4, /*oDim=*/4, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, G, c, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{G, F, c, h, w},
               /*outputMap=*/{N, G, F, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNgchwGfchwQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNgchwGfchwQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr G = m.dim(1);
  AffineExpr F = m.dim(2);
  AffineExpr H = m.dim(3);
  AffineExpr W = m.dim(4);
  AffineExpr c = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/0)
          .matchStride(/*iDim=*/4, /*fDim=*/4, /*oDim=*/4, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, G, c, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{G, F, c, h, w},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, G, F, H, W}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwgcGfhwcOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwgcGfhwcOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr G = m.dim(3);
  AffineExpr F = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);
  AffineExpr c = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/2, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/3, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), G, c},
               /*filterMap=*/{G, F, h, w, c},
               /*outputMap=*/{N, H, W, G, F}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv2DNhwgcGfhwcQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv2DNhwgcGfhwcQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr G = m.dim(3);
  AffineExpr F = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);
  AffineExpr c = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/2, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/3, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), G, c},
               /*filterMap=*/{G, F, h, w, c},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, H, W, G, F}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv3DOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (isa<linalg::Conv3DOp>(op)) {
    // Conv3DOp has no strides/dilations attributes, default to 1.
    result.dilations = SmallVector<int64_t>(3, 1);
    result.strides = SmallVector<int64_t>(3, 1);
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr D = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr d = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/0, /*fDim=*/0, /*oDim=*/0, /*idx=*/0)
          .matchStride(/*iDim=*/1, /*fDim=*/1, /*oDim=*/1, /*idx=*/1)
          .matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/2)
          .matchMaps({/*inputMap=*/{m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2)},
                      /*filterMap=*/{d, h, w},
                      /*outputMap=*/{D, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv3DNdhwcDhwcfOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv3DNdhwcDhwcfOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr F = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);
  AffineExpr c = m.dim(8);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), c},
                      /*filterMap=*/{d, h, w, c, F},
                      /*outputMap=*/{N, D, H, W, F}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv3DNdhwcDhwcfQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv3DNdhwcDhwcfQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr F = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);
  AffineExpr c = m.dim(8);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), c},
                      /*filterMap=*/{d, h, w, c, F},
                      /*scalarMap=*/{},
                      /*scalarMap=*/{},
                      /*outputMap=*/{N, D, H, W, F}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::Conv3DNcdhwFcdhwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp = dyn_cast<linalg::Conv3DNcdhwFcdhwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr F = m.dim(1);
  AffineExpr D = m.dim(2);
  AffineExpr H = m.dim(3);
  AffineExpr W = m.dim(4);
  AffineExpr c = m.dim(5);
  AffineExpr d = m.dim(6);
  AffineExpr h = m.dim(7);
  AffineExpr w = m.dim(8);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/2, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/3, /*oDim=*/3, /*idx=*/1)
          .matchStride(/*iDim=*/4, /*fDim=*/4, /*oDim=*/4, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, c, m.strided(D, d, 0),
                                    m.strided(H, h, 1), m.strided(W, w, 2)},
                      /*filterMap=*/{F, c, d, h, w},
                      /*outputMap=*/{N, F, D, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv1DNcwCwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv1DNcwCwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, C, m.strided(W, w, 0)},
                      /*filterMap=*/{C, w},
                      /*outputMap=*/{N, C, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv1DNwcWcOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w, C},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcmOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv1DNwcWcmOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr CM = m.dim(3);
  AffineExpr w = m.dim(4);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w, C, CM},
                      /*outputMap=*/{N, W, C, CM}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv2DNchwChwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv2DNchwChwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, C, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{C, h, w},
               /*outputMap=*/{N, C, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv2DNhwcHwcOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w, C},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv2DNhwcHwcQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w, C},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, H, W, C}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv2DNhwcHwcmOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr CM = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w, C, CM},
               /*outputMap=*/{N, H, W, C, CM}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmQOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv2DNhwcHwcmQOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr CM = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w, C, CM},
               /*scalarMap=*/{},
               /*scalarMap=*/{},
               /*outputMap=*/{N, H, W, C, CM}})
          .matchBody(/*containsZeroPointOffset=*/true))
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv3DNdhwcDhwcOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr d = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);
  AffineExpr C = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), C},
                      /*filterMap=*/{d, h, w, C},
                      /*outputMap=*/{N, D, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv3DNcdhwCdhwOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv3DNcdhwCdhwOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr d = m.dim(4);
  AffineExpr h = m.dim(5);
  AffineExpr w = m.dim(6);
  AffineExpr C = m.dim(7);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/1)
          .matchStride(/*iDim=*/4, /*fDim=*/3, /*oDim=*/4, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, C, m.strided(D, d, 0),
                                    m.strided(H, h, 1), m.strided(W, w, 2)},
                      /*filterMap=*/{C, d, h, w},
                      /*outputMap=*/{N, C, D, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcmOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto convOp =
          dyn_cast<linalg::DepthwiseConv3DNdhwcDhwcmOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr CM = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);
  AffineExpr C = m.dim(8);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), C},
                      /*filterMap=*/{d, h, w, C, CM},
                      /*outputMap=*/{N, D, H, W, C, CM}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNhwcMaxOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::MaxSigned);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNhwcMinOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNhwcMinOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::MinSigned);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNhwcSumOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNhwcSumOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::Sum);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp =
          dyn_cast<linalg::PoolingNhwcMaxUnsignedOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::MaxUnsigned);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp =
          dyn_cast<linalg::PoolingNhwcMinUnsignedOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::MinUnsigned);
  AffineExpr N = m.dim(0);
  AffineExpr H = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr C = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, m.strided(H, h, 0), m.strided(W, w, 1), C},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNchwSumOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNchwSumOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::Sum);
  AffineExpr N = m.dim(0);
  AffineExpr C = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/0, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/1, /*oDim=*/3, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, C, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, C, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNchwMaxOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNchwMaxOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/2, &result.dilations,
                       &result.strides, PoolingType::MaxSigned);
  AffineExpr N = m.dim(0);
  AffineExpr C = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr h = m.dim(4);
  AffineExpr w = m.dim(5);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/0, /*oDim=*/2, /*idx=*/0)
          .matchStride(/*iDim=*/3, /*fDim=*/1, /*oDim=*/3, /*idx=*/1)
          .matchMaps(
              {/*inputMap=*/{N, C, m.strided(H, h, 0), m.strided(W, w, 1)},
               /*filterMap=*/{h, w},
               /*outputMap=*/{N, C, H, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNwcSumOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNwcSumOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::Sum);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNcwSumOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNcwSumOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::Sum);
  AffineExpr N = m.dim(0);
  AffineExpr C = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/0, /*oDim=*/2, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, C, m.strided(W, w, 0)},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, C, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNwcMaxOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNwcMaxOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::MaxSigned);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNwcMaxUnsignedOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp =
          dyn_cast<linalg::PoolingNwcMaxUnsignedOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::MaxUnsigned);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNcwMaxOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNcwMaxOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::MaxSigned);
  AffineExpr N = m.dim(0);
  AffineExpr C = m.dim(1);
  AffineExpr W = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/2, /*fDim=*/0, /*oDim=*/2, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, C, m.strided(W, w, 0)},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, C, W}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNwcMinOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNwcMinOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::MinSigned);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNwcMinUnsignedOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp =
          dyn_cast<linalg::PoolingNwcMinUnsignedOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/1, &result.dilations,
                       &result.strides, PoolingType::MinUnsigned);
  AffineExpr N = m.dim(0);
  AffineExpr W = m.dim(1);
  AffineExpr C = m.dim(2);
  AffineExpr w = m.dim(3);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchMaps({/*inputMap=*/{N, m.strided(W, w, 0), C},
                      /*filterMap=*/{w},
                      /*outputMap=*/{N, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNdhwcSumOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNdhwcSumOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides, PoolingType::Sum);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr C = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), C},
                      /*filterMap=*/{d, h, w},
                      /*outputMap=*/{N, D, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNdhwcMaxOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNdhwcMaxOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides, PoolingType::MaxSigned);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr C = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), C},
                      /*filterMap=*/{d, h, w},
                      /*outputMap=*/{N, D, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

template <>
std::optional<DilationsAndStrides>
matchConvolutionOpOfType<linalg::PoolingNdhwcMinOp>(LinalgOp op) {
  DilationsAndStrides result;
  if (auto poolOp = dyn_cast<linalg::PoolingNdhwcMinOp>(op.getOperation())) {
    result.dilations =
        llvm::to_vector(poolOp.getDilations().getValues<int64_t>());
    result.strides = llvm::to_vector(poolOp.getStrides().getValues<int64_t>());
    return result;
  }

  if (!isaConvolutionOpInterface(op))
    return std::nullopt;

  ConvMatcherBuilder m(op, /*spatialRank=*/3, &result.dilations,
                       &result.strides, PoolingType::MinSigned);
  AffineExpr N = m.dim(0);
  AffineExpr D = m.dim(1);
  AffineExpr H = m.dim(2);
  AffineExpr W = m.dim(3);
  AffineExpr C = m.dim(4);
  AffineExpr d = m.dim(5);
  AffineExpr h = m.dim(6);
  AffineExpr w = m.dim(7);

  if (m.matchStride(/*iDim=*/1, /*fDim=*/0, /*oDim=*/1, /*idx=*/0)
          .matchStride(/*iDim=*/2, /*fDim=*/1, /*oDim=*/2, /*idx=*/1)
          .matchStride(/*iDim=*/3, /*fDim=*/2, /*oDim=*/3, /*idx=*/2)
          .matchMaps({/*inputMap=*/{N, m.strided(D, d, 0), m.strided(H, h, 1),
                                    m.strided(W, w, 2), C},
                      /*filterMap=*/{d, h, w},
                      /*outputMap=*/{N, D, H, W, C}})
          .matchBody())
    return result;
  return std::nullopt;
}

Value makeComposedPadHighOp(OpBuilder &b, Location loc, RankedTensorType type,
                            Value source, Value pad, bool nofold,
                            ValueRange typeDynDims) {
  // Exit if `source` is not defined by an ExtractSliceOp.
  auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Search the `source` use-def chain for padded LinalgOps.
  Value current = sliceOp.getSource();
  while (current) {
    auto linalgOp = current.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      break;
    OpResult opResult = cast<OpResult>(current);
    current = linalgOp.getDpsInitOperand(opResult.getResultNumber())->get();
  }
  auto padOp = current ? current.getDefiningOp<tensor::PadOp>() : nullptr;

  // Exit if the search fails to match a tensor::PadOp at the end of the matched
  // LinalgOp sequence.
  if (!padOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the padded result type does not match.
  if (sliceOp.getSource().getType() != type)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the LinalgOps are not high padded.
  if (llvm::any_of(padOp.getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) != static_cast<int64_t>(0);
      }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if `padOpSliceOp`, which defines the slice used by
  // `padOp`, is rank-reducing.
  auto padOpSliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!padOpSliceOp ||
      sliceOp.getMixedSizes().size() != padOpSliceOp.getMixedSizes().size())
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the sizes of the dynamic sizes of `sliceOp` do not match the size
  // of the slice padded by `padOp`.
  if (llvm::any_of(
          llvm::zip(sliceOp.getMixedSizes(), padOpSliceOp.getMixedSizes()),
          [](std::tuple<OpFoldResult, OpFoldResult> it) {
            return !isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it));
          }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the padding values do not match.
  Attribute padOpPadAttr, padAttr;
  Value padOpPad = padOp.getConstantPaddingValue();
  if (!padOpPad || !matchPattern(padOpPad, m_Constant(&padOpPadAttr)) ||
      !matchPattern(pad, m_Constant(&padAttr)) || padOpPadAttr != padAttr)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Return the padded result if the padding values and sizes match.
  return sliceOp.getSource();
}

GenericOp makeMemRefCopyOp(OpBuilder &b, Location loc, Value from, Value to) {
  auto memrefTypeTo = cast<MemRefType>(to.getType());
#ifndef NDEBUG
  auto memrefTypeFrom = cast<MemRefType>(from.getType());
  assert(memrefTypeFrom.getRank() == memrefTypeTo.getRank() &&
         "`from` and `to` memref must have the same rank");
#endif // NDEBUG

  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
      b, loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        linalg::YieldOp::create(b, loc, args.front());
      });
}

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected as many entries for proc info as number of loops, even if "
         "they are null entries");
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);
  LoopNest loopNest = mlir::scf::buildLoopNest(
      b, loc, lbs, ubs, steps, iterArgInitValues,
      [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
        assert(iterArgs.size() == iterArgInitValues.size() &&
               "expect the number of output tensors and iter args to match");
        SmallVector<Value> operandValuesToUse = linalgOp->getOperands();
        if (!iterArgs.empty()) {
          operandValuesToUse = linalgOp.getDpsInputs();
          operandValuesToUse.append(iterArgs.begin(), iterArgs.end());
        }
        return bodyBuilderFn(b, loc, ivs, operandValuesToUse);
      });

  if (loopNest.loops.empty() || procInfo.empty())
    return;

  // Filter out scf.for loops that were created out of parallel dimensions.
  for (const auto &loop : llvm::enumerate(loopNest.loops)) {
    if (procInfo[loop.index()].distributionMethod ==
        DistributionMethod::Cyclic) {
      mapLoopToProcessorIds(loop.value(), procInfo[loop.index()].procId,
                            procInfo[loop.index()].nprocs);
    }
  }
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> /*procInfo*/) {
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected AffineForOp init values");
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);

  // Affine loops require constant steps.
  SmallVector<int64_t, 4> constantSteps;
  constantSteps.reserve(steps.size());
  for (Value v : steps) {
    auto constVal = getConstantIntValue(v);
    assert(constVal.has_value() && "Affine loops require constant steps");
    constantSteps.push_back(constVal.value());
  }

  affine::buildAffineLoopNest(b, loc, lbs, ubs, constantSteps,
                              [&](OpBuilder &b, Location loc, ValueRange ivs) {
                                bodyBuilderFn(b, loc, ivs,
                                              linalgOp->getOperands());
                              });
}

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
void updateBoundsForCyclicDistribution(OpBuilder &b, Location loc, Value procId,
                                       Value nprocs, Value &lb, Value &ub,
                                       Value &step) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
  lb =
      affine::makeComposedAffineApply(b, loc, d0 + d1 * s0, {lb, procId, step});
  step = affine::makeComposedAffineApply(b, loc, d0 * s0, {nprocs, step});
}

/// Generates a loop nest consisting of scf.parallel and scf.for, depending
/// on the `iteratorTypes.` Consecutive parallel loops create a single
/// scf.parallel operation; each sequential loop creates a new scf.for
/// operation. The body of the innermost loop is populated by
/// `bodyBuilderFn` that accepts a range of induction variables for all
/// loops. `ivStorage` is used to store the partial list of induction
/// variables.
// TODO: this function can be made iterative instead. However, it
// will have at most as many recursive calls as nested loops, which rarely
// exceeds 10.
static void generateParallelLoopNest(
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<linalg::ProcInfo> procInfo,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn,
    SmallVectorImpl<Value> &ivStorage) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());
  assert(procInfo.empty() || (lbs.size() == procInfo.size()));

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty()) {
    bodyBuilderFn(b, loc, ivStorage);
    return;
  }

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse.
  if (!isParallelIterator(iteratorTypes.front())) {
    LoopNest singleLoop = buildLoopNest(
        b, loc, lbs.take_front(), ubs.take_front(), steps.take_front(),
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          ivStorage.append(ivs.begin(), ivs.end());
          generateParallelLoopNest(
              b, loc, lbs.drop_front(), ubs.drop_front(), steps.drop_front(),
              iteratorTypes.drop_front(),
              procInfo.empty() ? procInfo : procInfo.drop_front(),
              bodyBuilderFn, ivStorage);
        });
    return;
  }

  unsigned nLoops = iteratorTypes.size();
  unsigned numProcessed = 0;
  DistributionMethod distributionMethod = DistributionMethod::None;
  if (procInfo.empty()) {
    numProcessed = nLoops - iteratorTypes.drop_while(isParallelIterator).size();
  } else {
    distributionMethod = procInfo.front().distributionMethod;
    numProcessed =
        nLoops - procInfo
                     .drop_while([&](linalg::ProcInfo p) {
                       return p.distributionMethod == distributionMethod;
                     })
                     .size();
  }

  auto remainderProcInfo =
      procInfo.empty() ? procInfo : procInfo.drop_front(numProcessed);
  switch (distributionMethod) {
  case DistributionMethod::None: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    scf::ParallelOp::create(
        b, loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::Cyclic: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    scf::ParallelOp::create(
        b, loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::CyclicNumProcsGeNumIters: {
    // Check (for the processed loops) that the iteration is in-bounds.
    ArithBuilder ab(b, loc);
    Value cond = ab.slt(lbs[0], ubs[0]);
    for (unsigned i = 1; i < numProcessed; ++i)
      cond = ab._and(cond, ab.slt(lbs[i], ubs[i]));
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    scf::IfOp::create(b, loc, cond, [&](OpBuilder &b, Location loc) {
      generateParallelLoopNest(b, loc, lbs.drop_front(numProcessed),
                               ubs.drop_front(numProcessed),
                               steps.drop_front(numProcessed),
                               iteratorTypes.drop_front(numProcessed),
                               remainderProcInfo, bodyBuilderFn, ivStorage);
      scf::YieldOp::create(b, loc, ValueRange{});
    });
    return;
  }
  case DistributionMethod::CyclicNumProcsEqNumIters:
    // No check/loops needed here. Set the `%iv` to be the `%lb` and proceed
    // with inner loop generation.
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    generateParallelLoopNest(
        b, loc, lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
        steps.drop_front(numProcessed), iteratorTypes.drop_front(numProcessed),
        remainderProcInfo, bodyBuilderFn, ivStorage);
    return;
  }
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected proc information for all loops when present");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unsigned numLoops = iteratorTypes.size();
  ivs.reserve(numLoops);
  lbsStorage.reserve(numLoops);
  ubsStorage.reserve(numLoops);
  stepsStorage.reserve(numLoops);

  // Get the loop lb, ub, and step.
  unpackRanges(b, loc, loopRanges, lbsStorage, ubsStorage, stepsStorage);

  // Modify the lb, ub, and step based on the distribution options.
  for (const auto &it : llvm::enumerate(procInfo)) {
    if (it.value().distributionMethod != linalg::DistributionMethod::None) {
      updateBoundsForCyclicDistribution(
          b, loc, it.value().procId, it.value().nprocs, lbsStorage[it.index()],
          ubsStorage[it.index()], stepsStorage[it.index()]);
    }
  }
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  generateParallelLoopNest(
      b, loc, lbs, ubs, steps, iteratorTypes, procInfo,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        bodyBuilderFn(b, loc, ivs, linalgOp->getOperands());
      },
      ivs);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

static Operation *materializeTiledShape(OpBuilder &builder, Location loc,
                                        Value valueToTile,
                                        const SliceParameters &sliceParams) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&](MemRefType) {
                        return memref::SubViewOp::create(
                            builder, loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Case([&](RankedTensorType) {
                        return tensor::ExtractSliceOp::create(
                            builder, loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .DefaultUnreachable("Unexpected shaped type");
  return sliceOp;
}

Operation *makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                          ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                          ArrayRef<OpFoldResult> lbs,
                          ArrayRef<OpFoldResult> ubs,
                          ArrayRef<OpFoldResult> subShapeSizes,
                          bool omitPartialTileCheck) {
  SliceParameters sliceParams =
      computeSliceParameters(builder, loc, valueToTile, tileSizes, map, lbs,
                             ubs, subShapeSizes, omitPartialTileCheck);
  return materializeTiledShape(builder, loc, valueToTile, sliceParams);
}

SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Compute offsets/sizes/strides for the tile.
  SliceParameters sliceParams;
  sliceParams.offsets.reserve(rank);
  sliceParams.sizes.reserve(rank);
  sliceParams.strides.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      sliceParams.offsets.push_back(builder.getIndexAttr(0));
      OpFoldResult dim = createFoldedDimOp(builder, loc, valueToTile, r);
      sliceParams.sizes.push_back(dim);
      sliceParams.strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: submap: " << m << "\n");
    IRRewriter rewriter(builder);
    // The offset of the slice is m(lbs) - m(0).
    SmallVector<Attribute> zeros(lbs.size(), rewriter.getIndexAttr(0));
    SmallVector<Attribute> mAtZero;
    [[maybe_unused]] auto res = m.constantFold(zeros, mAtZero);
    assert(succeeded(res) && "affine_map must be evaluatable (not symbols)");
    int64_t mAtZeroInt =
        cast<IntegerAttr>(mAtZero[0]).getValue().getSExtValue();
    OpFoldResult offset = makeComposedFoldedAffineApply(
        rewriter, loc, m.getResult(0) - mAtZeroInt, lbs);
    sliceParams.offsets.push_back(offset);

    OpFoldResult closedIntSize =
        makeComposedFoldedAffineApply(rewriter, loc, m, subShapeSizes);
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    OpFoldResult size =
        makeComposedFoldedAffineApply(rewriter, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: new offset: " << offset << "\n");
    sliceParams.strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
      sliceParams.sizes.push_back(size);
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    std::optional<int64_t> sizeCst = getConstantIntValue(size);
    auto hasTileSizeOne = sizeCst == 1;
    auto dividesEvenly = sizeCst && ShapedType::isStatic(shapeSize) &&
                         ((shapeSize % *sizeCst) == 0);
    if (!hasTileSizeOne && !dividesEvenly) {
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: shapeSize=" << shapeSize
                              << ", size: " << size
                              << ": make sure in bound with affine.min\n");

      AffineExpr dim0, dim1, dim2;
      MLIRContext *context = builder.getContext();
      bindDims(context, dim0, dim1, dim2);

      // Get the dimension size for this dimension. We need to first calculate
      // the max index and then plus one. This is important because for
      // convolution ops, we have its input window dimension's affine map of the
      // form `(d0 * s0 + d1)`, where `d0`/`d1 is an output/filter window
      // dimension and `s0` is stride. Directly use the dimension size of
      // output/filer window dimensions will cause incorrect calculation.
      AffineMap minusOneMap = AffineMap::inferFromExprList(
                                  {ArrayRef<AffineExpr>{dim0 - 1}}, context)
                                  .front();
      AffineMap plusOneMap = AffineMap::inferFromExprList(
                                 {ArrayRef<AffineExpr>{dim0 + 1}}, context)
                                 .front();
      SmallVector<OpFoldResult> maxIndices =
          llvm::to_vector(llvm::map_range(ubs, [&](OpFoldResult ub) {
            return makeComposedFoldedAffineApply(rewriter, loc, minusOneMap,
                                                 {ub});
          }));
      OpFoldResult maxIndex =
          makeComposedFoldedAffineApply(rewriter, loc, m, maxIndices);
      OpFoldResult d =
          makeComposedFoldedAffineApply(rewriter, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}}, context)
                             .front();
      size =
          makeComposedFoldedAffineMin(rewriter, loc, minMap, {size, d, offset});
    }
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    sliceParams.sizes.push_back(size);
  }
  return sliceParams;
}

SmallVector<OpFoldResult> computeTileOffsets(OpBuilder &b, Location loc,
                                             ArrayRef<OpFoldResult> ivs,
                                             ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> offsets;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for loop#" << idx << "\n");
    bool isTiled = !isZeroInteger(tileSizes[idx]);
    offsets.push_back(isTiled ? ivs[idxIvs++] : b.getIndexAttr(0));
    LLVM_DEBUG(llvm::dbgs()
               << "computeTileOffsets: " << offsets.back() << "\n");
  }
  return offsets;
}

SmallVector<OpFoldResult> computeTileSizes(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           ArrayRef<OpFoldResult> sizeBounds) {
  SmallVector<OpFoldResult> sizes;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZeroInteger(tileSizes[idx]);
    // Before composing, we need to make range a closed interval.
    OpFoldResult size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    IRRewriter rewriter(b);
    sizes.push_back(makeComposedFoldedAffineApply(rewriter, loc, d0 - 1, size));
    LLVM_DEBUG(llvm::dbgs() << "computeTileSizes: " << sizes.back() << "\n");
  }
  return sizes;
}

SmallVector<Type> getTensorOutputTypes(LinalgOp op, ValueRange operands) {
  if (op.hasPureBufferSemantics())
    return {};
  return llvm::to_vector(
      llvm::map_range(op.getDpsInitsMutable(), [&](OpOperand &opOperand) {
        return operands[opOperand.getOperandNumber()].getType();
      }));
}

SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    LinalgOp op, ValueRange operands,
                                    ValueRange results) {
  if (op.hasPureBufferSemantics())
    return {};
  SmallVector<Value> tensorResults;
  tensorResults.reserve(results.size());
  // Insert a insert_slice for each output tensor.
  unsigned resultIdx = 0;
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    // TODO: use an interface/adaptor to avoid leaking position in
    // `tiledOperands`.
    Value outputTensor = operands[opOperand.getOperandNumber()];
    if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
      Value inserted = tensor::InsertSliceOp::create(
          builder, loc, sliceOp.getSource().getType(), results[resultIdx],
          sliceOp.getSource(), sliceOp.getOffsets(), sliceOp.getSizes(),
          sliceOp.getStrides(), sliceOp.getStaticOffsets(),
          sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
      tensorResults.push_back(inserted);
    } else {
      tensorResults.push_back(results[resultIdx]);
    }
    ++resultIdx;
  }
  return tensorResults;
}

SmallVector<std::optional<SliceParameters>>
computeAllSliceParameters(OpBuilder &builder, Location loc, LinalgOp linalgOp,
                          ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                          ArrayRef<OpFoldResult> tileSizes,
                          ArrayRef<OpFoldResult> sizeBounds,
                          bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroInteger(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             linalgOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<std::optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  for (auto [opOperand, val] :
       llvm::zip(linalgOp->getOpOperands(), valuesToTile)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getMatchingIndexingMap(&opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) && !(isa<RankedTensorType>(operandType) &&
                                      linalgOp.isDpsInit(&opOperand))) {
      allSliceParams.push_back(std::nullopt);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

SmallVector<Value> makeTiledShapes(OpBuilder &builder, Location loc,
                                   LinalgOp linalgOp, ValueRange valuesToTile,
                                   ArrayRef<OpFoldResult> ivs,
                                   ArrayRef<OpFoldResult> tileSizes,
                                   ArrayRef<OpFoldResult> sizeBounds,
                                   bool omitPartialTileCheck) {
  SmallVector<std::optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, linalgOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
                  ->getResult(0)
            : valueToTile);
  }
  return tiledShapes;
}

void offsetIndices(OpBuilder &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  IRRewriter rewriter(b);
  offsetIndices(rewriter, linalgOp, offsets);
}

void offsetIndices(RewriterBase &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  if (!linalgOp.hasIndexSemantics())
    return;

  for (IndexOp indexOp : linalgOp.getBlock()->getOps<IndexOp>()) {
    if (indexOp.getDim() >= offsets.size() || !offsets[indexOp.getDim()])
      continue;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(indexOp);
    AffineExpr index, offset;
    bindDims(b.getContext(), index, offset);
    OpFoldResult applied = makeComposedFoldedAffineApply(
        b, indexOp.getLoc(), index + offset,
        {getAsOpFoldResult(indexOp.getResult()), offsets[indexOp.getDim()]});
    Value materialized =
        getValueOrCreateConstantIndexOp(b, indexOp.getLoc(), applied);
    b.replaceUsesWithIf(indexOp, materialized, [&](OpOperand &use) {
      return use.getOwner() != materialized.getDefiningOp();
    });
  }
}

/// Get the reassociation maps to fold the result of a extract_slice (or source
/// of a insert_slice) operation with given offsets, and sizes to its
/// rank-reduced version. This is only done for the cases where the size is 1
/// and offset is 0. Strictly speaking the offset 0 is not required in general,
/// but non-zero offsets are not handled by SPIR-V backend at this point (and
/// potentially cannot be handled).
std::optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices curr;
  for (const auto &it : llvm::enumerate(mixedSizes)) {
    auto dim = it.index();
    auto size = it.value();
    curr.push_back(dim);
    auto attr = llvm::dyn_cast_if_present<Attribute>(size);
    if (attr && cast<IntegerAttr>(attr).getInt() == 1)
      continue;
    reassociation.emplace_back(ReassociationIndices{});
    std::swap(reassociation.back(), curr);
  }
  // When the reassociations are not empty, then fold the remaining
  // unit-dimensions into the last dimension.  If the reassociations so far is
  // empty, then leave it emtpy. This will fold everything to a rank-0 tensor.
  if (!curr.empty() && !reassociation.empty())
    reassociation.back().append(curr.begin(), curr.end());
  return reassociation;
}

} // namespace linalg
} // namespace mlir
