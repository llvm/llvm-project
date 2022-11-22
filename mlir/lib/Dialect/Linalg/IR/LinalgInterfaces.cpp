//===- LinalgInterfaces.cpp - Linalg interfaces implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::linalg;

/// Include the definitions of the copy operation interface.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Interface utility functions
//===----------------------------------------------------------------------===//
bool linalg::detail::canOpOperandsBeDroppedImpl(
    linalg::LinalgOp linalgOp, ArrayRef<OpOperand *> droppedOperands) {
  SmallVector<AffineMap> indexingMaps;
  for (auto &opOperand : linalgOp->getOpOperands()) {
    if (llvm::is_contained(droppedOperands, &opOperand))
      continue;
    indexingMaps.push_back(linalgOp.getMatchingIndexingMap(&opOperand));
  }
  if (indexingMaps.empty()) {
    // If there are no indexing maps, the operand can only be dropped
    // if the op has no loops.
    return linalgOp.getNumLoops() == 0;
  }
  return inversePermutation(concatAffineMaps(indexingMaps)) != AffineMap();
}

//===----------------------------------------------------------------------===//
// ContractionOpInterface implementation
//===----------------------------------------------------------------------===//

/// Return true if the use-def chain from `v` to `from` consists of 0 or more
/// unary single-operand operations.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static bool isChainOfUnaryOpsFrom(Value v, Value from) {
  while (true) {
    if (v == from)
      return true;
    Operation *op = v.getDefiningOp();
    if (!op || op->getNumOperands() != 1)
      return false;
    v = op->getOperand(0);
  };
}

/// Return the unique instance of OpType in `block` if it is indeed unique.
/// Return null if none or more than 1 instances exist.
template <typename OpType>
static OpType getSingleOpOfType(Block &block) {
  OpType res = nullptr;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

/// Detect whether res is any permutation of `u5(u1(c) + u2(u3(a) * u4(b)))`
/// on the field (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent
/// unary operations that may change the type.
template <typename AddOpType, typename MulOpType>
static bool isAddMul(Block &block) {
  if (block.getNumArguments() != 3)
    return false;
  Operation *yieldOp = block.getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;

  AddOpType addOp = getSingleOpOfType<AddOpType>(block);
  MulOpType mulOp = getSingleOpOfType<MulOpType>(block);
  if (!addOp || !mulOp)
    return false;

  Value argA = block.getArgument(0), argB = block.getArgument(1);
  Value a = mulOp->getOperand(0), b = mulOp->getOperand(1);
  Value mul = mulOp->getResult(0);
  Value argC = block.getArgument(2);
  Value c1 = addOp->getOperand(0), c2 = addOp->getOperand(1);
  Value add = addOp->getResult(0);
  Value res = yieldOp->getOperand(0);
  // Result traces back to add.
  auto un = isChainOfUnaryOpsFrom;
  bool success = un(res, add);
  // One of the operands of add traces back to argC, the other to the mul.
  success |= (un(c1, argC) && un(c2, mul)) || ((un(c1, mul)) && un(c2, argC));
  // One of the operands of mul traces back to argA, the other to argB.
  success |= (un(a, argA) && un(b, argB)) || ((un(a, argB)) && un(b, argA));
  return success;
}

enum class MatchContractionResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  NoReduction,
  NotProjectedPermutations,
  NotAddMul
};
static MatchContractionResult isContractionInterfaceImpl(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchContractionResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
    return MatchContractionResult::WrongNumOperands;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (linalgOp.getNumReductionLoops() == 0)
    return MatchContractionResult::NoReduction;
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); }))
    return MatchContractionResult::NotProjectedPermutations;
  // TODO: more fields than add/mul.
  if (!isAddMul<arith::AddFOp, arith::MulFOp>(linalgOp->getRegion(0).front()) &&
      !isAddMul<arith::AddIOp, arith::MulIOp>(linalgOp->getRegion(0).front()) &&
      !isAddMul<complex::AddOp, complex::MulOp>(
          linalgOp->getRegion(0).front()) &&
      !isAddMul<arith::OrIOp, arith::AndIOp>(linalgOp->getRegion(0).front()))
    return MatchContractionResult::NotAddMul;
  return MatchContractionResult::Success;
}

bool mlir::linalg::isaContractionOpInterface(LinalgOp linalgOp) {
  if (!linalgOp)
    return false;
  Operation *op = linalgOp.getOperation();
  return isa<ContractionOpInterface>(op) ||
         (isContractionInterfaceImpl(op) == MatchContractionResult::Success);
}

/// Verify that a LinalgOp `op` is a contraction.
/// A Linalg contraction is defined in general terms:
///   1. Has 2 input and 1 output shapes.
///   2. Has at least one reduction dimension.
///   3. Has only projected permutation indexing maps.
///   4. its body computes `u5(u1(c) + u2(u3(a) * u4(b)))` on some field
///   (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent scalar unary
///   operations that may change the type (e.g. for mixed-precision).
/// As a consequence, when vectorization of such an op occurs, the only special
/// behavior is that the (unique) MulOpType is vectorized into a
/// `vector.contract`. All other ops are handled in a generic fashion.
/// In the future, we may wish to allow more input arguments and elementwise and
/// constant operations that do not involve the reduction dimension(s).
LogicalResult mlir::linalg::detail::verifyContractionInterface(Operation *op) {
  auto res = isContractionInterfaceImpl(op);
  if (res == MatchContractionResult::NotLinalgOp)
    return op->emitError("expected a LinalgOp");
  if (res == MatchContractionResult::WrongNumOperands)
    return op->emitError("expected op with 2 inputs and 1 outputs");
  if (res == MatchContractionResult::NoReduction)
    return op->emitError("expected at least a reduction loop");
  if (res == MatchContractionResult::NotProjectedPermutations)
    return op->emitError("expected all indexings to be projected permutations");
  if (res == MatchContractionResult::NotAddMul)
    return op->emitError("(add, mul) operations not found");
  return success();
}

//===----------------------------------------------------------------------===//
// ConvolutionOpInterface implementation
//===----------------------------------------------------------------------===//

/// Of the given two expressions returns one that is of type T (`lhs` gets
/// preference over `rhs`)
template <typename T>
static T getAffineExprOfType(AffineExpr lhs, AffineExpr rhs) {
  return lhs.isa<T>() ? lhs.cast<T>()
                      : (rhs.isa<T>() ? rhs.cast<T>() : nullptr);
}

namespace {
/// Walk the indexing expressions for input of a convolution operation to verify
/// its of the right form, either
/// - AffineDimExpr
/// - AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?
///      (`+` AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?)*
///
/// classifies the AffineDimExpr as convolved dimensions or unconvolved
/// dimensions and verifies each dimension occurs only once.
struct ConvAccessExprWalker
    : public AffineExprVisitor<ConvAccessExprWalker, LogicalResult> {
  llvm::SmallDenseSet<unsigned> convolvedDims;
  llvm::SmallDenseSet<unsigned> unConvolvedDims;

  LogicalResult visitDimExpr(AffineDimExpr dimExpr) {
    unsigned position = dimExpr.getPosition();
    if (unConvolvedDims.count(position) || convolvedDims.count(position)) {
      return failure();
    }
    unConvolvedDims.insert(position);
    return success();
  }

  LogicalResult visitSymbolExpr(AffineSymbolExpr expr) { return failure(); }

  LogicalResult visitConstantExpr(AffineConstantExpr expr) { return failure(); }

  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryExpr) {
    // In pre-order visit, top level op has to be an add op.
    if (binaryExpr.getKind() != AffineExprKind::Add)
      return failure();
    return success(succeeded(isDimExprOrMulExpr(binaryExpr.getLHS())) &&
                   succeeded(isDimExprOrMulExpr(binaryExpr.getRHS())));
  }

  LogicalResult isDimExprOrMulExpr(AffineExpr expr) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      unsigned dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      convolvedDims.insert(dim);
      return success();
    }
    if (auto symbolMulExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
      if (symbolMulExpr.getKind() != AffineExprKind::Mul)
        return failure();
      auto lhsExpr = symbolMulExpr.getLHS();
      auto rhsExpr = symbolMulExpr.getRHS();
      // Check for symbol expression.
      AffineExpr mulExpr =
          getAffineExprOfType<AffineSymbolExpr>(lhsExpr, rhsExpr);
      // If there was no symbol expr, check for constant expression.
      if (!mulExpr) {
        mulExpr = getAffineExprOfType<AffineConstantExpr>(lhsExpr, rhsExpr);
      }
      auto dimExpr = getAffineExprOfType<AffineDimExpr>(lhsExpr, rhsExpr);
      if (!mulExpr || !dimExpr)
        return failure();
      unsigned dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      convolvedDims.insert(dim);
      return success();
    }
    return failure();
  }
};
} // namespace

static llvm::SmallDenseSet<unsigned> getPreservedDims(AffineMap map) {
  assert(map.isProjectedPermutation() &&
         "expected map to have projected permutations");
  llvm::SmallDenseSet<unsigned> preservedDims;
  for (auto expr : map.getResults())
    preservedDims.insert(expr.cast<AffineDimExpr>().getPosition());
  return preservedDims;
}

enum class MatchConvolutionResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  WrongInputIndexingMap,
  NotProjectedPermutations,
  NonConvolutionLoop,
  OutputDimsNotParallel,
  NonOutputDimNotReduction
};

static MatchConvolutionResult isConvolutionInterfaceImpl(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchConvolutionResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() < 2 || linalgOp.getNumDpsInits() != 1)
    return MatchConvolutionResult::WrongNumOperands;

  auto indexingMaps = linalgOp.getIndexingMapsArray();

  // Check the input indexing map has the right form.
  ConvAccessExprWalker inputExprWalker;
  if (llvm::any_of(indexingMaps[0].getResults(),
                   [&inputExprWalker](AffineExpr expr) {
                     return failed(inputExprWalker.visit(expr));
                   })) {
    return MatchConvolutionResult::WrongInputIndexingMap;
  }

  // Filter and output maps must be projected permutation.
  if (!indexingMaps[1].isProjectedPermutation() ||
      !indexingMaps.back().isProjectedPermutation())
    return MatchConvolutionResult::NotProjectedPermutations;

  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  llvm::SmallDenseSet<unsigned> outputDims =
      getPreservedDims(indexingMaps.back());
  llvm::SmallDenseSet<unsigned> filterDims = getPreservedDims(indexingMaps[1]);
  // Make sure all loops are charecterized as one of:
  // - Batch loop : present in output, as non-convolved in input, not present in
  //   filter.
  // - Output image dimension : present in output, convolved dims in input, not
  //   present in filter.
  // - Output channel dimension : present in output, not present in input,
  //   present in filter.
  // - Filter loop dimension : present in filter, convolved in input, not
  //   present in output.
  // - Input channel dimension : unconvolved in input, not present in output,
  //   present in filter.
  // - Depth multiplier : unconvolved in input, present in output, present in
  //   filter.
  llvm::SmallDenseSet<unsigned> allLoopDims;
  for (auto outputExpr : indexingMaps.back().getResults()) {
    unsigned outputDim = outputExpr.cast<AffineDimExpr>().getPosition();
    if (inputExprWalker.unConvolvedDims.count(outputDim) &&
        !filterDims.count(outputDim)) {
      // Batch dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (inputExprWalker.convolvedDims.count(outputDim) &&
        !filterDims.count(outputDim)) {
      // Output image Loop dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (!inputExprWalker.convolvedDims.count(outputDim) &&
        !inputExprWalker.unConvolvedDims.count(outputDim) &&
        filterDims.count(outputDim)) {
      // Output channel dimension.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(outputDim) &&
        filterDims.count(outputDim)) {
      // Depth multiplier.
      if (iteratorTypes[outputDim] != utils::IteratorType::parallel)
        return MatchConvolutionResult::OutputDimsNotParallel;
      allLoopDims.insert(outputDim);
      continue;
    }
    return MatchConvolutionResult::NonConvolutionLoop;
  }
  for (auto filterExpr : indexingMaps[1].getResults()) {
    unsigned filterDim = filterExpr.cast<AffineDimExpr>().getPosition();
    if (outputDims.count(filterDim) &&
        !inputExprWalker.unConvolvedDims.count(filterDim) &&
        !inputExprWalker.convolvedDims.count(filterDim)) {
      // Output channel dimension. THis is already seen, continue;
      continue;
    }
    if (inputExprWalker.convolvedDims.count(filterDim) &&
        !outputDims.count(filterDim)) {
      // Filter loop dimension.
      if (iteratorTypes[filterDim] != utils::IteratorType::reduction)
        return MatchConvolutionResult::NonOutputDimNotReduction;
      if (allLoopDims.count(filterDim))
        return MatchConvolutionResult::NonConvolutionLoop;
      allLoopDims.insert(filterDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(filterDim) &&
        !outputDims.count(filterDim)) {
      // Input channel dimension.
      if (iteratorTypes[filterDim] != utils::IteratorType::reduction)
        return MatchConvolutionResult::NonOutputDimNotReduction;
      if (allLoopDims.count(filterDim))
        return MatchConvolutionResult::NonConvolutionLoop;
      allLoopDims.insert(filterDim);
      continue;
    }
    if (inputExprWalker.unConvolvedDims.count(filterDim) &&
        outputDims.count(filterDim)) {
      // Depthwise loop. Already seen.
      continue;
    }
    return MatchConvolutionResult::NonConvolutionLoop;
  }
  // All loops must be covered now.
  if (allLoopDims.size() != linalgOp.getNumLoops())
    return MatchConvolutionResult::NonConvolutionLoop;

  return MatchConvolutionResult::Success;
}

LogicalResult mlir::linalg::detail::verifyConvolutionInterface(Operation *op) {
  auto res = isConvolutionInterfaceImpl(op);
  if (res == MatchConvolutionResult::NotLinalgOp)
    return op->emitError("expected a LinalgOp");
  if (res == MatchConvolutionResult::WrongNumOperands)
    return op->emitError("expected op with 2 inputs and 1 output");
  if (res == MatchConvolutionResult::WrongInputIndexingMap)
    return op->emitError("unexpected input index map for convolutions");
  if (res == MatchConvolutionResult::NotProjectedPermutations) {
    return op->emitError(
        "expected output/filter indexing maps to be projected permutations");
  }
  if (res == MatchConvolutionResult::NonConvolutionLoop) {
    return op->emitError("unexpected loop dimension for convolution op");
  }
  if (res == MatchConvolutionResult::OutputDimsNotParallel) {
    return op->emitError(
        "expected all iterators used to access outputs to be parallel");
  }
  if (res == MatchConvolutionResult::NonOutputDimNotReduction) {
    return op->emitError(
        "expected all iterators not used to access outputs to be reduction");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FillOpInterface implementation
//===----------------------------------------------------------------------===//

enum class MatchFillResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  NotScalarInput
};

static MatchFillResult isFillInterfaceImpl(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return MatchFillResult::NotLinalgOp;
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return MatchFillResult::WrongNumOperands;

  OpOperand *value = linalgOp.getDpsInputOperand(0);
  if (!linalgOp.isScalar(value))
    return MatchFillResult::NotScalarInput;

  return MatchFillResult::Success;
}

LogicalResult mlir::linalg::detail::verifyFillInterface(Operation *op) {
  auto res = isFillInterfaceImpl(op);
  if (res == MatchFillResult::NotLinalgOp)
    return op->emitError("expected a LinalgOp");
  if (res == MatchFillResult::WrongNumOperands)
    return op->emitError("expected op with 1 input and 1 output");
  if (res == MatchFillResult::NotScalarInput)
    return op->emitError("expected op with scalar input");

  return success();
}

//===----------------------------------------------------------------------===//
// StructuredOpInterface implementation
//===----------------------------------------------------------------------===//

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
static Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source,
                               int64_t dim) {
  if (source.getType().isa<UnrankedMemRefType, MemRefType>())
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (source.getType().isa<UnrankedTensorType, RankedTensorType>())
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}
static OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value source,
                                      int64_t dim) {
  auto shapedType = source.getType().cast<ShapedType>();
  if (!shapedType.hasRank() || shapedType.isDynamicDim(dim))
    return createOrFoldDimOp(b, loc, source, dim);
  return b.getIndexAttr(shapedType.getDimSize(dim));
}

SmallVector<OpFoldResult> LinalgOp::createFlatListOfOperandDims(OpBuilder &b,
                                                                Location loc) {
  SmallVector<OpFoldResult> res;
  for (OpOperand &opOperand : getOperation()->getOpOperands()) {
    for (int64_t i = 0, e = getRank(&opOperand); i < e; ++i)
      res.push_back(createFoldedDimOp(b, loc, opOperand.get(), i));
  }
  return res;
}

SmallVector<int64_t, 4> LinalgOp::createFlatListOfOperandStaticDims() {
  SmallVector<int64_t, 4> res;
  assert(!hasDynamicShape() && "expected operands to have static shapes");
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    llvm::append_range(res, getShape(&opOperand));
  return res;
}

SmallVector<Range, 4> LinalgOp::createLoopRanges(OpBuilder &b, Location loc) {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  auto viewSizes = createFlatListOfOperandDims(b, loc);
  SmallVector<Range, 4> res(numDims);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>()) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] =
          Range{b.getIndexAttr(0), viewSizes[idx], b.getIndexAttr(1)};
    }
  }
  return res;
}

SmallVector<int64_t, 4> LinalgOp::computeStaticLoopSizes() {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes = createFlatListOfOperandStaticDims();
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>())
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}

/// Visitor to check if any of the given set of positions from AffineDimExprs
/// are used within an AffineExpr.
struct HasAffineDimExprVisitor
    : public AffineExprVisitor<HasAffineDimExprVisitor, bool> {
  HasAffineDimExprVisitor(llvm::SmallBitVector positions)
      : positions(std::move(positions)) {}

  bool visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryOpExpr) {
    return visit(binaryOpExpr.getLHS()) || visit(binaryOpExpr.getRHS());
  }

  bool visitDimExpr(AffineDimExpr dimExpr) {
    return positions.test(dimExpr.getPosition());
  }

  bool visitConstantExpr(AffineConstantExpr constExpr) { return false; }

  bool visitSymbolExpr(AffineSymbolExpr symbolExpr) { return false; }

private:
  llvm::SmallBitVector positions;
};

static std::pair<int64_t, int64_t>
getResultsPositionInLoopsToShapeMap(LinalgOp &op) {
  int64_t inputRankSum = 0;
  int64_t outputRankSum = 0;
  for (OpOperand *input : op.getDpsInputOperands())
    inputRankSum += op.getRank(input);
  for (OpOperand *output : op.getDpsInitOperands())
    outputRankSum += op.getRank(output);
  return {inputRankSum, inputRankSum + outputRankSum};
}

LogicalResult
LinalgOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  // An example that helps understand the logic below.
  // Consider the following expression O(i+j, j) += A(i,k) * B(k, j)
  // We want to express the shape of dim 0 of O in terms of shape of the inputs.
  // This is achieved as follows.
  //   loopsToShapesMap = (d0, d1, d2) -> (d0, d2, d2, d1, d0 + d1, d1)
  //   subMapOfResultShapes = (d0, d1, d2) -> (d0 + d1, d1)
  //   shapesToLoopsMap = (d0, d2, d2, d3, d4, d5) -> (d0, d3, d2)
  //   resultShapesFromInputShapes = subMapOfResultDim.compose(shapesToLoopMap)
  //     = (d0, d1, d2, d3, d4, d5) -> (d0 + d1, d1)
  AffineMap loopsToShapesMap = getLoopsToShapesMap();

  // Find the position in the above map that represents the shape of the
  // result:dim being inferred.
  auto resultShapesSubMapPos = getResultsPositionInLoopsToShapeMap(*this);

  /// From loopsToShapesMap extract the submap that represents the shape of the
  /// (resultIdx, dim) needed.
  AffineMap loopToResultsShapeMap = loopsToShapesMap.getSliceMap(
      resultShapesSubMapPos.first,
      resultShapesSubMapPos.second - resultShapesSubMapPos.first);
  AffineMap resultShapesFromInputShapesMap =
      loopToResultsShapeMap.compose(getShapesToLoopsMap());

  // Check that the result dim map does not contain the positions corresponding
  // to the outputs.
  llvm::SmallBitVector outputDims(resultShapesFromInputShapesMap.getNumDims());
  outputDims.set(resultShapesSubMapPos.first, resultShapesSubMapPos.second);
  HasAffineDimExprVisitor checkDimExpr(std::move(outputDims));
  Location loc = getOperation()->getLoc();
  IRRewriter rewriter(b);
  SmallVector<OpFoldResult> allResultDimValues =
      makeComposedFoldedMultiResultAffineApply(
          rewriter, loc, resultShapesFromInputShapesMap,
          createFlatListOfOperandDims(b, loc));
  int64_t pos = 0;
  ArrayRef<AffineExpr> shapeExprs = resultShapesFromInputShapesMap.getResults();
  for (OpOperand *opOperand : getDpsInitOperands()) {
    SmallVector<Value> shapes;
    for (int64_t dim : llvm::seq<int64_t>(0, getRank(opOperand))) {
      if (checkDimExpr.visit(shapeExprs[pos]))
        shapes.push_back(createOrFoldDimOp(b, loc, opOperand->get(), dim));
      else
        shapes.push_back(
            getValueOrCreateConstantIndexOp(b, loc, allResultDimValues[pos]));
      pos++;
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
  }
  return success();
}

LogicalResult mlir::linalg::detail::verifyStructuredOpInterface(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);

  // Before checking indexing maps, we need to make sure the attributes
  // referenced by it are valid.
  if (linalgOp.hasDynamicIndexingMaps())
    if (failed(linalgOp.verifyIndexingMapRequiredAttributes()))
      return failure();

  // All input/output operands must be indexed.
  if (static_cast<int64_t>(linalgOp.getIndexingMapsArray().size()) !=
      linalgOp->getNumOperands())
    return op->emitOpError("expected the number of indexing_map (")
           << linalgOp.getIndexingMapsArray().size()
           << ") to be equal to the number of input/output operands ("
           << linalgOp->getNumOperands() << ")";

  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);

    // Symbols disallowed.
    if (indexingMap.getNumSymbols() != 0)
      return op->emitOpError("unexpected symbols in indexing_map #")
             << opOperand.getOperandNumber();

    // Domain must be consistent.
    unsigned numLoops = linalgOp.getNumLoops();
    if (indexingMap.getNumDims() != numLoops)
      return op->emitOpError("expected indexing_map #")
             << opOperand.getOperandNumber() << " to have " << numLoops
             << " dim(s) to match the number of loops";

    int64_t rank = linalgOp.getRank(&opOperand);
    if (indexingMap.getNumResults() != rank)
      return op->emitOpError("expected operand rank (")
             << rank << ") to match the result rank of indexing_map #"
             << opOperand.getOperandNumber() << " ("
             << indexingMap.getNumResults() << ")";
  }

  SmallVector<unsigned> redDims;
  linalgOp.getReductionDims(redDims);

  if (!linalgOp.getShapesToLoopsMap())
    return op->emitOpError("expected the shape-to-loops map to be non-null");

  // Check if given shapes match to inferred shapes.
  SmallVector<int64_t, 4> endLoopRangeValues = linalgOp.getStaticLoopRanges();
  SmallVector<int64_t, 4> startLoopRangeValues(endLoopRangeValues.size(), 0);

  // Verify only static cases since we can't get exact dimension sizes and loop
  // ranges for dynamic cases in this stage.
  if (llvm::none_of(endLoopRangeValues, ShapedType::isDynamic)) {
    for (int64_t &range : endLoopRangeValues)
      range -= 1;
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
      SmallVector<int64_t, 4> startIndices =
          indexingMap.compose(startLoopRangeValues);
      SmallVector<int64_t, 4> endIndices =
          indexingMap.compose(endLoopRangeValues);
      ArrayRef<int64_t> shape = linalgOp.getShape(&opOperand);
      for (auto dim : llvm::seq<int64_t>(0, shape.size())) {
        // Ignore dynamic dimension or the case that the dimension size is 0
        if (ShapedType::isDynamic(shape[dim]) || shape[dim] == 0)
          continue;

        // The first index or last index should be the maximum or the minimum in
        // the inferred index ranges since the range is increasing or
        // decreasing. The size of dimensions of input/output operands and the
        // maximum value + 1 in the inferred range should be the same. But, for
        // now we check if the inferred ranges are in boundary of input/output
        // operands' size or not in case that Affine Expressions are complicated
        // such as d0 * 3
        // + d1 since it is not easy to handle the issues.
        // Found the case that this solution can't check, for example, (d0, d1)
        // -> (d1 - d0)
        int64_t inferredDimSize =
            std::max(startIndices[dim], endIndices[dim]) + 1;
        if (std::min(startIndices[dim], endIndices[dim]) < 0) {
          std::string mapStr;
          {
            llvm::raw_string_ostream os(mapStr);
            os << indexingMap;
          }
          return op->emitOpError(
                     "unexpected result less than 0 at expression #")
                 << dim << " in " << mapStr;
        }
        if (indexingMap.getResult(dim).dyn_cast<AffineDimExpr>()) {
          if (inferredDimSize != shape[dim]) {
            return op->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be " << inferredDimSize << ", but found "
                   << shape[dim];
          }
        } else {
          if (inferredDimSize > shape[dim]) {
            return op->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be greater than or equal to "
                   << inferredDimSize << ", but found " << shape[dim];
          }
        }
      }
    }
  }

  // Check the region has exactly one block.
  if (linalgOp->getNumRegions() != 1 ||
      !llvm::hasSingleElement(linalgOp->getRegion(0)))
    return op->emitOpError("expects to have 1 region with 1 block");

  // Simplifying assumption: bbargs match 1-1 with shape operands elemental
  // types.
  // TODO: once ranked shape types are plugged in, we may want to drop the
  // corresponding bbargs, that can never be read from. This will be subject to
  // consistency discussions (i.e. what to do with output tensors whose bbarg is
  // not used).
  Block &block = linalgOp->getRegion(0).front();

  if (linalgOp.getOpOperandsMatchingBBargs().size() != block.getNumArguments())
    return op->emitOpError("expected as many non-induction variable region "
                           "arguments as the number of input/output operands");

  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    Type elementType = getElementTypeOrSelf(opOperand->get());
    Type argType = block.getArgument(opOperand->getOperandNumber()).getType();
    if (elementType != argType)
      return op->emitOpError("expected type of bb argument #")
             << opOperand->getOperandNumber() << " (" << argType << ")"
             << " to match element or self type of the corresponding operand ("
             << elementType << ")";
  }

  return success();
}
