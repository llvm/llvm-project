//===- DecomposeLinalgOps.cpp - Pattern to break up Linalg ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Pattern to decompose a GenericOp that has more than two statements
/// into one GenericOp with the first statement (i.e. peeled operation), and
/// a second GenericOp with the remaining statements (i.e. residual operations).

/// - The result of the first GenericOp has the same shape as the iteration
///   space of the GenericOp. The body of the op yields as many values as the
///   original op plus all the results of the peeled operation.
/// - The second GenericOp has as many operands as the original operation plus
/// all the results of the first Generic Op. It has the same number of yields as
/// the original op.
/// - If the result of the peeled operation was yielded by the original
///   GenericOp the uses of the corresponding results will be replaced with the
///   result of the first GenericOp created.
///
///  Example
///
/// ```mlir
///  %result:2 = linalg.generic ... ins(%arg0, %arg1, %arg2 : ...)
///      outs(%init0, %init1 : ...) {
///    ^bb0(%b0: ... , %b1: ... , %b2: ... , %b3: ..., %b4: ...):
///      %0 = <s0> %b0, %b1 : ...
///      %1 = <s1> %0, %b2 : ...
///      linalg.yield %0, %1 : ...
///  } -> (..., ...)
///  return %result#0, %result#1
/// ```
///
/// gets split into
///
/// ```mlir
/// %init = tensor.empty ...
/// %op0:3 = linalg.generic ... ins(%arg0, %arg1, %arg2 : ...)
///      outs(%init0, %init1, %init : ...)
///    ^bb0(%b0: ... , %b1: ... , %b2: ... , %b3: ..., %b4: ..., %b5: ...):
///      %0 = <s0> %b0, %b1 : ...
///      linalg.yield %0, %..., %0 : ...
///  } -> (..., ..., ...)
/// %op1:2 = linalg.generic ... ins(%arg0, %arg1, %arg2, %op0#2 : ...)
///      outs(%init0, %init1 : ...) {
///    ^bb0(%b0: ... , %b1: ... , %b2: ... , %b3: ..., %b4: ..., %b5: ...):
///      %1 = <s1> %b3, %b2 : ...
///      linalg.yield %..., %1 : ...
///  } -> (..., ...)
///  return %op0#0, %op1#1
/// ```
///
/// After canonicalization this is expected to be
///
/// ```mlir
/// %init = tensor.empty ...
/// %op0 = linalg.generic ... ins(%arg0, %arg1, : ...)
///      outs(%init : ...)
///    ^bb0(%b0: ... , %b1: ... , %b2: ...):
///      %0 = <s0> %b0, %b1 : ...
///      linalg.yield %0 : ...
///  } -> ...
/// %op1 = linalg.generic ... ins(%arg2, %op0#2 : ...)
///      outs(%init1 : ...) {
///    ^bb0(%b0: ... , %b1: ... , %b2: ...):
///      %1 = <s1> %b1, %b0 : ...
///      linalg.yield %..., %1 : ...
///  } -> ...
///  return %op0, %op1
/// ```
struct DecomposeLinalgOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override;

private:
  /// Helper method to create a generic op for the peeled scalar operation. The
  /// created op has an empty region.
  GenericOp createPeeledGenericOp(GenericOp genericOp,
                                  PatternRewriter &rewriter) const;

  /// Helper method to create a generic op for the residual scalar operation.
  /// The created op has the same region as the original op.
  GenericOp createResidualGenericOp(GenericOp genericOp,
                                    GenericOp peeledGenericOp,
                                    PatternRewriter &rewriter) const;
};
} // namespace

/// Helper method to compute the range of a generic op.
static SmallVector<OpFoldResult> getGenericOpLoopRange(OpBuilder &b,
                                                       GenericOp op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  auto allShapesSizes =
      cast<LinalgOp>(op.getOperation()).createFlatListOfOperandDims(b, loc);
  AffineMap map = op.getShapesToLoopsMap();
  IRRewriter rewriter(b);
  return affine::makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                          allShapesSizes);
}

/// Helper method to permute the list of `values` based on the `map`.
SmallVector<OpFoldResult> permuteValues(ArrayRef<OpFoldResult> values,
                                        AffineMap map) {
  assert(map.isPermutation());
  SmallVector<OpFoldResult> permutedValues(values.size());
  for (const auto &position :
       llvm::enumerate(llvm::map_range(map.getResults(), [](AffineExpr expr) {
         return cast<AffineDimExpr>(expr).getPosition();
       })))
    permutedValues[position.value()] = values[position.index()];
  return permutedValues;
}

/// Get zero value for an element type.
static Value getZero(OpBuilder &b, Location loc, Type elementType) {
  assert(elementType.isIntOrIndexOrFloat() &&
         "expected scalar type while computing zero value");
  if (isa<IntegerType>(elementType))
    return b.create<arith::ConstantIntOp>(loc, 0, elementType);
  if (elementType.isIndex())
    return b.create<arith::ConstantIndexOp>(loc, 0);
  // Assume float.
  auto floatType = cast<FloatType>(elementType);
  return b.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
}

GenericOp
DecomposeLinalgOp::createPeeledGenericOp(GenericOp genericOp,
                                         PatternRewriter &rewriter) const {
  Block *body = genericOp.getBody();
  Operation *peeledScalarOperation = &(*body->begin());
  SmallVector<AffineMap> peeledGenericOpIndexingMaps =
      genericOp.getIndexingMapsArray();

  /// Compute the loop ranges for operation. This is the shape of the result of
  /// the generic op for the peeled operation.
  Location loc = genericOp.getLoc();
  SmallVector<OpFoldResult> domain = getGenericOpLoopRange(rewriter, genericOp);
  SmallVector<Value> newInitValues;
  SmallVector<Type> newResultTypes;

  // Add as many new results as the number of results of the peeled scalar op.
  for (auto scalarOpResult : peeledScalarOperation->getResults()) {
    // If the result is yielded by the original op, use the operand, indexing
    // map and result type that correspond to the yielded value.

    std::optional<unsigned> resultNumber;
    for (auto *user : scalarOpResult.getUsers()) {
      if (auto yieldOp = dyn_cast<YieldOp>(user)) {
        // Find the first use of the `scalarOpResult` in the yield op.
        for (OpOperand &yieldOperand : yieldOp->getOpOperands()) {
          if (yieldOperand.get() == scalarOpResult) {
            resultNumber = yieldOperand.getOperandNumber();
            break;
          }
        }
        assert(resultNumber && "unable to find use of a value in its user");
        break;
      }
    }
    if (resultNumber) {
      newInitValues.push_back(
          genericOp.getDpsInitOperand(*resultNumber)->get());
      OpResult result = cast<OpResult>(genericOp.getResult(*resultNumber));
      newResultTypes.push_back(result.getType());
      peeledGenericOpIndexingMaps.push_back(
          genericOp.getIndexingMapMatchingResult(result));
      continue;
    }

    // Fall back path, use an `init_tensor` and identity indexing map.
    AffineMap indexingMap = rewriter.getMultiDimIdentityMap(domain.size());
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, domain, scalarOpResult.getType());
    newInitValues.push_back(emptyTensor);
    newResultTypes.push_back(emptyTensor.getType());
    peeledGenericOpIndexingMaps.push_back(indexingMap);
  }

  /// Create the peeled generic op with an empty body.
  SmallVector<Value> outsOperands = genericOp.getOutputs();
  outsOperands.append(newInitValues.begin(), newInitValues.end());
  SmallVector<Type> resultTypes = llvm::to_vector(genericOp.getResultTypes());
  resultTypes.append(newResultTypes.begin(), newResultTypes.end());
  auto indexingMapAttr =
      rewriter.getAffineMapArrayAttr(peeledGenericOpIndexingMaps);
  return rewriter.create<GenericOp>(
      loc, resultTypes, genericOp.getInputs(), outsOperands, indexingMapAttr,
      genericOp.getIteratorTypes(), /*doc=*/nullptr, /*libraryCall=*/nullptr,
      [](OpBuilder, Location, ValueRange) {});
}

GenericOp
DecomposeLinalgOp::createResidualGenericOp(GenericOp genericOp,
                                           GenericOp peeledGenericOp,
                                           PatternRewriter &rewriter) const {
  /// Append all results from the peeledGenericOps as `ins` operand for the
  /// residual generic op.
  SmallVector<Value> residualGenericOpOperands = genericOp.getInputs();
  unsigned origNumResults = genericOp.getNumResults();
  unsigned peeledGenericOpNumResults = peeledGenericOp.getNumResults();
  SmallVector<Value> extraIns;
  for (auto resultNum :
       llvm::seq<unsigned>(origNumResults, peeledGenericOpNumResults))
    extraIns.push_back(peeledGenericOp->getResult(resultNum));
  residualGenericOpOperands.append(extraIns);

  /// Add indexing maps for the newly added operands. Use the same map
  /// as those used for the new results of the peeledGenericOp.
  auto indexingMaps = llvm::to_vector(
      llvm::map_range(genericOp.getDpsInputOperands(), [&](OpOperand *operand) {
        return genericOp.getMatchingIndexingMap(operand);
      }));
  for (auto resultNum :
       llvm::seq<unsigned>(origNumResults, peeledGenericOpNumResults)) {
    OpResult result = cast<OpResult>(peeledGenericOp.getResult(resultNum));
    indexingMaps.push_back(
        peeledGenericOp.getIndexingMapMatchingResult(result));
  }
  for (OpOperand &outOperand : genericOp.getDpsInitsMutable())
    indexingMaps.push_back(genericOp.getMatchingIndexingMap(&outOperand));

  auto indexingMapAttr = rewriter.getAffineMapArrayAttr(indexingMaps);
  return rewriter.create<GenericOp>(
      genericOp->getLoc(), genericOp->getResultTypes(),
      residualGenericOpOperands, genericOp.getOutputs(), indexingMapAttr,
      genericOp.getIteratorTypes(), /*doc=*/nullptr, /*libraryCall=*/nullptr,
      [](OpBuilder, Location, ValueRange) {});
}

LogicalResult
DecomposeLinalgOp::matchAndRewrite(GenericOp genericOp,
                                   PatternRewriter &rewriter) const {
  /// For now only match on operations where the iterator types are all parallel
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "unhandled decomposition of operation "
                                       "with non-parallel iterator types");
  }
  // TODO: this could be generalized to handle `linalg.generic` with buffer
  // operands too but requires allocation for intermediates. Punt on this for
  // now.
  if (!genericOp.hasPureTensorSemantics()) {
    return rewriter.notifyMatchFailure(
        genericOp, "only operations with tensor semantics are handled");
  }

  if (llvm::any_of(genericOp.getDpsInitsMutable(), [&](OpOperand &outOperand) {
        return !genericOp.getMatchingIndexingMap(&outOperand).isPermutation();
      })) {
    return rewriter.notifyMatchFailure(
        genericOp, "unhandled decomposition of generic op with out operand not "
                   "accessed using a permutation");
  }

  /// If the op has only a single statement (apart from the yield), do nothing.
  Block *body = genericOp.getBody();
  if (body->getOperations().size() <= 2) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "operation has less than 3 statements");
  }

  /// Check that the peeled statement has a scalar element type.
  if (llvm::any_of(body->getOperations().begin()->getResultTypes(),
                   [](Type t) { return !t.isIntOrIndexOrFloat(); })) {
    return rewriter.notifyMatchFailure(
        &(*body->getOperations().begin()),
        "expected return type to be only int, index or float");
  }

  GenericOp peeledGenericOp = createPeeledGenericOp(genericOp, rewriter);
  GenericOp residualGenericOp =
      createResidualGenericOp(genericOp, peeledGenericOp, rewriter);

  /// Move the first statement of the original operation into the body of the
  /// generic op for the peeled operation.
  Block *peeledGenericOpBody = peeledGenericOp.getBody();
  Block *residualGenericOpBody = residualGenericOp.getBody();
  assert(peeledGenericOpBody->empty() && residualGenericOpBody->empty() &&
         "expected split generic ops to have empty region");
  peeledGenericOpBody->getOperations().splice(
      peeledGenericOpBody->begin(), body->getOperations(), body->begin());
  residualGenericOpBody->getOperations().splice(residualGenericOpBody->begin(),
                                                body->getOperations());

  Operation *peeledScalarOperation = &(*peeledGenericOpBody->begin());
  auto *yieldOp = residualGenericOpBody->getTerminator();
  {
    // Yield all the result of the peeled scalar operation.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(peeledGenericOpBody);
    SmallVector<Value> yieldedVals;
    for (auto origYield : yieldOp->getOperands()) {
      if (origYield.getDefiningOp() == peeledScalarOperation) {
        yieldedVals.push_back(origYield);
      } else {
        // Do not materialize any new ops inside of the decomposed LinalgOp,
        // as that would trigger another application of the rewrite pattern
        // (infinite loop).
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(peeledGenericOp);
        yieldedVals.push_back(
            getZero(rewriter, genericOp.getLoc(), origYield.getType()));
      }
    }
    yieldedVals.append(llvm::to_vector(
        llvm::map_range(peeledScalarOperation->getResults(),
                        [](OpResult opr) -> Value { return opr; })));
    rewriter.create<YieldOp>(genericOp.getLoc(), yieldedVals);
  }

  /// In the split operations, replace block arguments uses that refer to
  /// original operation to the block arguments of the newly created operation.
  unsigned origNumInputs = genericOp.getNumDpsInputs();
  for (const auto &inputBlockArg :
       llvm::enumerate(genericOp.getBody()->getArguments())) {
    Value residualOpReplacementArg =
        residualGenericOpBody->getArgument(inputBlockArg.index());
    rewriter.replaceUsesWithIf(
        inputBlockArg.value(), residualOpReplacementArg, [&](OpOperand &use) {
          return use.getOwner()->getBlock() == residualGenericOpBody;
        });

    Value peeledOpReplacementArg =
        peeledGenericOpBody->getArgument(inputBlockArg.index());
    rewriter.replaceUsesWithIf(
        inputBlockArg.value(), peeledOpReplacementArg, [&](OpOperand &use) {
          return use.getOwner()->getBlock() == peeledGenericOpBody;
        });
  }

  /// Before fixing up the residual operation, track what values are yielded. If
  /// any of those are from the peeled scalar operation, the uses of the
  /// corresponding result have to be remapped to result of the generic op for
  /// the peeled operation.
  SmallVector<Value> replacements;
  for (const auto &yieldValue : llvm::enumerate(yieldOp->getOperands())) {
    OpResult opr = dyn_cast<OpResult>(yieldValue.value());
    if (!opr || opr.getOwner() != peeledScalarOperation)
      replacements.push_back(residualGenericOp.getResult(yieldValue.index()));
    else
      replacements.push_back(peeledGenericOp->getResult(yieldValue.index()));
  }

  /// Update all uses of the peeled scalar operation results in the residual op
  /// to the newly added arguments.
  {
    SmallVector<Value> scalarReplacements;
    unsigned peeledScalarOpNumResults = peeledScalarOperation->getNumResults();
    scalarReplacements.reserve(peeledScalarOpNumResults);
    for (auto num : llvm::seq<unsigned>(0, peeledScalarOpNumResults))
      scalarReplacements.push_back(
          residualGenericOpBody->getArgument(num + origNumInputs));
    bool allUsesReplaced = false;
    rewriter.replaceOpWithinBlock(peeledScalarOperation, scalarReplacements,
                                  residualGenericOpBody, &allUsesReplaced);
    assert(!allUsesReplaced &&
           "peeled scalar operation is erased when it wasnt expected to be");
  }

  // Replace the original operation
  rewriter.replaceOp(genericOp, replacements);
  return success();
}

void mlir::linalg::populateDecomposeLinalgOpsPattern(
    RewritePatternSet &patterns, bool removeDeadArgsAndResults) {
  patterns.insert<DecomposeLinalgOp>(patterns.getContext());
  // Add the patterns to clean up the dead operands and results.
  if (removeDeadArgsAndResults)
    populateEraseUnusedOperandsAndResultsPatterns(patterns);
}
