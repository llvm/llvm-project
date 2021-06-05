//===- LinalgInterfaces.cpp - Linalg interfaces implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::linalg;

/// Include the definitions of the copy operation interface.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.cpp.inc"

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

enum MatchContractionResult {
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
  if (linalgOp.getNumInputs() != 2 || linalgOp.getNumOutputs() != 1)
    return MatchContractionResult::WrongNumOperands;
  auto mapRange = linalgOp.indexing_maps().getAsValueRange<AffineMapAttr>();
  if (linalgOp.getNumReductionLoops() == 0)
    return MatchContractionResult::NoReduction;
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); }))
    return MatchContractionResult::NotProjectedPermutations;
  // TODO: more fields than add/mul.
  if (!isAddMul<AddFOp, MulFOp>(linalgOp->getRegion(0).front()) &&
      !isAddMul<AddIOp, MulIOp>(linalgOp->getRegion(0).front()))
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
// StructuredOpInterface implementation
//===----------------------------------------------------------------------===//

OpOperandVector::operator SmallVector<Value>() {
  SmallVector<Value> result;
  result.reserve(this->size());
  llvm::transform(*this, std::back_inserter(result),
                  [](OpOperand *opOperand) { return opOperand->get(); });
  return result;
}

/// Fully compose map with operands and canonicalize the result.
/// Return the `createOrFold`'ed AffineApply op.
static Value createFoldedComposedAffineApply(OpBuilder &b, Location loc,
                                             AffineMap map,
                                             ValueRange operandsRef) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return b.createOrFold<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ValueRange values) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims(), numSym = map.getNumSymbols();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, numSym, expr);
    res.push_back(createFoldedComposedAffineApply(b, loc, map, values));
  }
  return res;
}

SmallVector<Value, 4> LinalgOp::createFlatListOfOperandDims(OpBuilder &b,
                                                            Location loc) {
  SmallVector<Value, 4> res;
  for (OpOperand *opOperand : getInputAndOutputOperands()) {
    for (int64_t i = 0, e = getRank(opOperand); i < e; ++i)
      res.push_back(b.createOrFold<memref::DimOp>(loc, opOperand->get(), i));
  }
  return res;
}

SmallVector<int64_t, 4> LinalgOp::createFlatListOfOperandStaticDims() {
  SmallVector<int64_t, 4> res;
  assert(!hasDynamicShape() && "expected operands to have static shapes");
  for (OpOperand *opOperand : getInputAndOutputOperands())
    llvm::append_range(res, getShape(opOperand));
  return res;
}

SmallVector<Range, 4> LinalgOp::createLoopRanges(OpBuilder &b, Location loc) {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  auto viewSizes = createFlatListOfOperandDims(b, loc);
  SmallVector<Range, 4> res(numDims);
  Value zeroVal = b.create<ConstantIndexOp>(loc, 0);
  Value oneVal = b.create<ConstantIndexOp>(loc, 1);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>()) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] = Range{zeroVal, viewSizes[idx], oneVal};
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
  HasAffineDimExprVisitor(llvm::SmallSet<unsigned, 4> &positions)
      : positions(positions) {}

  bool visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryOpExpr) {
    return visit(binaryOpExpr.getLHS()) || visit(binaryOpExpr.getRHS());
  }

  bool visitDimExpr(AffineDimExpr dimExpr) {
    return positions.count(dimExpr.getPosition());
  }

  bool visitConstantExpr(AffineConstantExpr constExpr) { return false; }

  bool visitSymbolExpr(AffineSymbolExpr symbolExpr) { return false; }

private:
  llvm::SmallSet<unsigned, 4> positions;
};

LogicalResult LinalgOp::reifyReturnTypeShapesPerResultDim(
    OpBuilder &b, SmallVectorImpl<SmallVector<Value>> &reifiedReturnShapes) {
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
  auto resultShapesSubMapPos = getResultsPositionInLoopsToShapeMap();

  /// From loopsToShapesMap extract the submap that represents the shape of the
  /// (resultIdx, dim) needed.
  SmallVector<unsigned, 4> resultPosRange =
      llvm::to_vector<4>(llvm::seq<unsigned>(resultShapesSubMapPos.first,
                                             resultShapesSubMapPos.second));
  AffineMap loopToResultsShapeMap = loopsToShapesMap.getSubMap(resultPosRange);
  AffineMap resultShapesFromInputShapesMap =
      loopToResultsShapeMap.compose(getShapesToLoopsMap());

  // Check that the result dim map does not contain the positions corresponding
  // to the outputs.
  llvm::SmallSet<unsigned, 4> outputDims;
  llvm::for_each(resultPosRange,
                 [&outputDims](unsigned dim) { outputDims.insert(dim); });
  HasAffineDimExprVisitor checkDimExpr(outputDims);
  Location loc = getOperation()->getLoc();
  auto allResultDimValues =
      applyMapToValues(b, loc, resultShapesFromInputShapesMap,
                       createFlatListOfOperandDims(b, loc));
  int64_t pos = 0;
  ArrayRef<AffineExpr> shapeExprs = resultShapesFromInputShapesMap.getResults();
  for (OpOperand *opOperand : getOutputOperands()) {
    SmallVector<Value> shapes;
    for (int64_t dim : llvm::seq<int64_t>(0, getRank(opOperand))) {
      if (checkDimExpr.visit(shapeExprs[pos]))
        shapes.push_back(
            b.createOrFold<memref::DimOp>(loc, opOperand->get(), dim));
      else
        shapes.push_back(allResultDimValues[pos]);
      pos++;
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
  }
  return success();
}

LogicalResult mlir::linalg::detail::verifyStructuredOpInterface(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);
  // Expect at least one input/output operand.
  // This means an op that constructs a tensor out of indices cannot be a
  // LinalgOp at the moment. For now this will have to be a special op until we
  // have output shape operands that are not tensors.
  int64_t numInputsAndOutputs = linalgOp.getNumInputsAndOutputs();
  if (numInputsAndOutputs == 0)
    return op->emitOpError("expected at least one input/output operand");
  if (failed(OpTrait::impl::verifyAtLeastNOperands(op, numInputsAndOutputs)))
    return failure();
  // Should have at least one output tensor per result tensor.
  // Can also have outbut buffers that do not correspond to results.
  if (op->getNumResults() > linalgOp.getOutputTensorOperands().size())
    return op->emitOpError("unexpected #results > #outputs");

  // Before checking indexing maps, we need to make sure the attributes
  // referenced by it are valid.
  if (linalgOp.hasDynamicIndexingMaps())
    if (failed(linalgOp.verifyIndexingMapRequiredAttributes()))
      return failure();

  // All shaped operands must be indexed.
  if (static_cast<int64_t>(linalgOp.indexing_maps().size()) !=
      linalgOp.getNumInputsAndOutputs())
    return op->emitOpError("expected the number of indexing_map (")
           << linalgOp.indexing_maps().size()
           << ") to be equal to the number of input/output operands ("
           << linalgOp.getNumInputsAndOutputs() << ")";

  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    AffineMap indexingMap = linalgOp.getTiedIndexingMap(opOperand);

    // Symbols disallowed.
    if (indexingMap.getNumSymbols() != 0)
      return op->emitOpError("unexpected symbols in indexing_map #")
             << opOperand->getOperandNumber();

    // Domain must be consistent.
    unsigned numLoops = linalgOp.getNumLoops();
    if (indexingMap.getNumDims() != numLoops)
      return op->emitOpError("expected indexing_map #")
             << opOperand->getOperandNumber() << " to have " << numLoops
             << " dim(s) to match the number of loops";

    int64_t rank = linalgOp.getRank(opOperand);
    if (indexingMap.getNumResults() != rank)
      return op->emitOpError("expected shaped value rank (")
             << rank << ") to match the result rank of indexing_map #"
             << opOperand->getOperandNumber() << " ("
             << indexingMap.getNumResults() << ")";
  }

  SmallVector<AffineExpr> redDims;
  linalgOp.getReductionDims(redDims);

  // Simplifying assumption: either full tensor or full buffer mode.
  // This allows simpler verification of output operands vs result types
  // without premature tracking of which operand is what in mixed-mode.
  // TODO: relax when mixed-mode needs to pass verification.
  if (!linalgOp.getOutputBufferOperands().empty() &&
      !linalgOp.getOutputTensorOperands().empty())
    return op->emitOpError(
        "expected output operands to all have tensor type or "
        "all have buffer type");

  for (OpOperand *opOperand : linalgOp.getOutputTensorOperands()) {
    // TODO: Enforce one output tensor per result?
    if (opOperand->getOperandNumber() - linalgOp.getNumInputs() >=
        linalgOp->getNumResults())
      continue;
    OpResult result = linalgOp.getTiedOpResult(opOperand);
    if (result.getType() != opOperand->get().getType())
      return op->emitOpError("expected type of operand #")
             << opOperand->getOperandNumber() << " ("
             << opOperand->get().getType() << ")"
             << " to match type of corresponding result (" << result.getType()
             << ")";
  }

  // Output tensor indexing map may not depend on reduction indices.
  for (OpOperand *opOperand : linalgOp.getOutputOperands()) {
    AffineMap indexingMap = linalgOp.getTiedIndexingMap(opOperand);
    for (auto expr : indexingMap.getResults()) {
      for (auto dim : redDims) {
        unsigned pos = dim.cast<AffineDimExpr>().getPosition();
        if (expr.isFunctionOfDim(pos)) {
          std::string exprStr;
          {
            llvm::raw_string_ostream os(exprStr);
            os << expr;
          }
          return op->emitOpError(
                     "unexpected output tensor expression in indexing map #")
                 << (opOperand->getOperandNumber() - linalgOp.getNumInputs())
                 << " a.k.a '" << exprStr
                 << "' is function of reduction iterator 'd" << pos << "'";
        }
      }
    }
  }

  // Named ops that are defined manually have a region builder but no region at
  // this time. Assume the region is well-formed by specification.
  // TODO: use linalg-ods-gen for all ops when we have enough expressive power.
  if (linalgOp->getNumRegions() == 0) {
    assert(!linalgOp.getRegionBuilder() && "regionBuilder but no region");
    return success();
  }

  auto &region = linalgOp->getRegion(0);
  if (linalgOp->getNumRegions() > 1 || !llvm::hasSingleElement(region))
    return op->emitOpError("expected 1 region with 1 block");

  if (!linalgOp.getShapesToLoopsMap())
    return op->emitOpError("expected the shape-to-loops map to be non-null");

  // Simplifying assumption: bbargs match 1-1 with shape operands elemental
  // types.
  // TODO: once ranked shape types are plugged in, we may want to drop the
  // corresponding bbargs, that can never be read from. This will be subject to
  // consistency discussions (i.e. what to do with output tensors whose bbarg is
  // not used).
  Block &block = linalgOp->getRegion(0).front();
  unsigned numBBIvs = linalgOp.getNumPayloadInductionVariables();

  if (linalgOp.getNumInputsAndOutputs() + numBBIvs != block.getNumArguments())
    return op->emitOpError("expected as many non-induction variable region "
                           "arguments as the number of shaped operands");

  // Note: the number and type of yield values are checked in the YieldOp.
  for (unsigned i = 0; i < numBBIvs; ++i)
    if (!block.getArgument(i).getType().isIndex())
      return op->emitOpError("expected index block argument #") << i;

  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    Type elementType = getElementTypeOrSelf(opOperand->get().getType());
    Type argType =
        block.getArgument(numBBIvs + opOperand->getOperandNumber()).getType();
    if (elementType != argType)
      return op->emitOpError("expected type of bb argument #")
             << numBBIvs + opOperand->getOperandNumber() << " (" << argType
             << ")"
             << " to match element type of corresponding shaped operand ("
             << elementType << ")";
  }

  // Check if given shapes match to inferred shapes.
  Optional<SmallVector<int64_t, 4>> endLoopRangeValues =
      linalgOp.getStaticLoopRanges();
  if (!endLoopRangeValues)
    return op->emitOpError("unable to find loop range for operation");
  SmallVector<int64_t, 4> startLoopRangeValues((*endLoopRangeValues).size(), 0);

  // Verify only static cases since we can't get exact dimension sizes and loop
  // ranges for dynamic cases in this stage.
  if (llvm::none_of(*endLoopRangeValues, ShapedType::isDynamic)) {
    for (int64_t &range : *endLoopRangeValues)
      range -= 1;
    for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
      AffineMap indexingMap = linalgOp.getTiedIndexingMap(opOperand);
      SmallVector<int64_t, 4> startIndices =
          indexingMap.compose(startLoopRangeValues);
      SmallVector<int64_t, 4> endIndices =
          indexingMap.compose(*endLoopRangeValues);
      ArrayRef<int64_t> shape = linalgOp.getShape(opOperand);
      for (auto dim : llvm::seq<int64_t>(0, shape.size())) {
        // Ignore dynamic dimension or the case that the dimension size is 0
        if (ShapedType::isDynamic(shape[dim]) || shape[dim] == 0)
          continue;

        // The first index or last index should be the maximum or the minimum in
        // the inferred index ranges since the range is increasing or
        // decreasing. The size of dimensions of shaped operands and the maximum
        // value + 1 in the inferred range should be the same. But, for now we
        // check if the inferred ranges are in boundary of shaped operands' size
        // or not in case that Affine Expressions are complicated such as d0 * 3
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
            return op->emitOpError("inferred shaped operand #")
                   << opOperand->getOperandNumber()
                   << " has shape's dimension #" << dim << " to be "
                   << inferredDimSize << ", but found " << shape[dim];
          }
        } else {
          if (inferredDimSize > shape[dim]) {
            return op->emitOpError("inferred shaped operand #")
                   << opOperand->getOperandNumber()
                   << " has shape's dimension #" << dim
                   << " to be greater than or equal to " << inferredDimSize
                   << ", but found " << shape[dim];
          }
        }
      }
    }
  }

  return success();
}
