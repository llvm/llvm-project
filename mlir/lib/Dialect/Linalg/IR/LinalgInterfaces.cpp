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
  for (Value v : getShapedOperands()) {
    ShapedType t = v.getType().template cast<ShapedType>();
    for (unsigned i = 0, e = t.getRank(); i < e; ++i)
      res.push_back(b.create<memref::DimOp>(loc, v, i));
  }
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

Optional<Value> LinalgOp::inferResultDimFromInputShapes(OpBuilder &b,
                                                        Location loc,
                                                        unsigned resultIdx,
                                                        unsigned dim) {
  // An example that helps understand the logic below.
  // Consider the following expression O(i+j, j) += A(i,k) * B(k, j)
  // We want to express the shape of dim 0 of O in terms of shape of the inputs.
  // This is achieved as follows.
  //   loopsToShapesMap = (d0, d1, d2) -> (d0, d2, d2, d1, d0 + d1, d1)
  //   subMapOfResultDim = (d0, d1, d2) -> (d0 + d1)
  //   shapesToLoopsMap = (d0, d2, d2, d3, d4, d5) -> (d0, d3, d2)
  //   resultFromFromInputDim = subMapOfResultDim.compose(shapesToLoopMap)
  //     = (d0, d1, d2, d3, d4, d5) -> (d0 + d1)
  AffineMap loopsToShapesMap = getLoopsToShapesMap();

  // Find the position in the above map that represents the shape of the
  // result:dim being inferred.
  Optional<unsigned> resultDimSubMapPos =
      getResultValueDimPositionInLoopsToShapeMap(resultIdx, dim);
  if (!resultDimSubMapPos)
    return {};

  /// From loopsToShapesMap extract the submap that represents the shape of the
  /// (resultIdx, dim) needed
  AffineMap loopToResultDimShapeMap =
      loopsToShapesMap.getSubMap(*resultDimSubMapPos);
  AffineMap operandShapesToResultDimMap =
      loopToResultDimShapeMap.compose(getShapesToLoopsMap());

  // Check that the result dim map does not contain the positions corresponding
  // to the outputs.
  llvm::SmallSet<unsigned, 4> outputDims;
  unsigned outputDimPosStart =
      getResultValueDimPositionInLoopsToShapeMap(0, 0).getValue();
  unsigned outputDimPosEnd =
      getResultValueDimPositionInLoopsToShapeMap(getNumOutputs() - 1,
                                                 getOutputOpOperands()
                                                         .back()
                                                         .get()
                                                         .getType()
                                                         .cast<ShapedType>()
                                                         .getRank() -
                                                     1)
          .getValue();
  llvm::for_each(llvm::seq<unsigned>(outputDimPosStart, outputDimPosEnd),
                 [&outputDims](unsigned dim) { outputDims.insert(dim); });
  HasAffineDimExprVisitor checkDimExpr(outputDims);
  if (checkDimExpr.visit(operandShapesToResultDimMap.getResult(0)))
    return llvm::None;
  return applyMapToValues(b, loc, operandShapesToResultDimMap,
                          createFlatListOfOperandDims(b, loc))[0];
}

LogicalResult mlir::linalg::detail::verifyStructuredOpInterface(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);
  // Expect at least one shaped operand.
  // This means an op that constructs a tensor out of indices cannot be a
  // LinalgOp at the moment. For now this will have to be a special op until we
  // have output shape operands that are not tensors.
  auto nShapedOperands = linalgOp.getNumShapedOperands();
  if (nShapedOperands == 0)
    return linalgOp.emitOpError("expected at least 1 Shaped operand");
  if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nShapedOperands)))
    return failure();
  // Should have at least one output tensor per result tensor.
  // Can also have outbut buffers that do not correspond to results.
  if (op->getNumResults() > linalgOp.getNumOutputTensors())
    return op->emitError("unexpected #results > #outputs");

  // Before checking indexing maps, we need to make sure the attributes
  // referenced by it are valid.
  if (linalgOp.hasDynamicIndexingMaps())
    if (failed(linalgOp.verifyIndexingMapRequiredAttributes()))
      return failure();

  // All shaped operands must be indexed.
  if (linalgOp.indexing_maps().size() != linalgOp.getNumShapedOperands())
    return linalgOp.emitOpError("expected the number of indexing_map (")
           << linalgOp.indexing_maps().size()
           << ") to be equal to the number of shaped operands ("
           << linalgOp.getNumShapedOperands() << ")";

  SmallVector<AffineMap, 4> indexingMaps;
  indexingMaps.reserve(linalgOp.indexing_maps().size());
  for (auto en : llvm::enumerate(linalgOp.indexing_maps())) {
    auto idx = en.index();
    auto m = en.value().template cast<AffineMapAttr>().getValue();
    indexingMaps.push_back(m); // Save reference to map for further checks.
    auto shapedValue = linalgOp.getShapedType(idx);

    // Symbols disallowed.
    if (m.getNumSymbols() != 0)
      return linalgOp.emitOpError("unexpected symbols in indexing_map #")
             << idx;

    // Domain must be consistent.
    auto nLoops = linalgOp.getNumLoops();
    if (m.getNumDims() != nLoops)
      return linalgOp.emitOpError("expected indexing_map #")
             << idx << " to have " << nLoops
             << " dim(s) to match the number of loops";

    if (m.getNumResults() != shapedValue.getRank())
      return linalgOp.emitOpError("expected shaped value rank (")
             << shapedValue.getRank()
             << ") to match the result rank of indexing_map #" << idx << " ("
             << m.getNumResults() << ")";
  }

  SmallVector<AffineExpr, 4> redDims;
  linalgOp.getReductionDims(redDims);

  // Simplifying assumption: either full tensor or full buffer mode.
  // This allows simpler verification of output operands vs result types
  // without premature tracking of which operand is what in mixed-mode.
  // TODO: relax when mixed-mode needs to pass verification.
  if (linalgOp.getNumOutputBuffers() > 0 && linalgOp.getNumOutputTensors() > 0)
    return op->emitError("expected output operands to all have tensor type or "
                         "all have buffer type");

  for (auto it :
       llvm::zip(linalgOp.getOutputOpOperands(), op->getResultTypes())) {
    if (!std::get<0>(it).get().getType().isa<RankedTensorType>())
      continue;
    if (std::get<0>(it).get().getType() != std::get<1>(it))
      return op->emitError("expected type of operand #")
             << std::get<0>(it).getOperandNumber() << " ("
             << std::get<0>(it).get().getType() << ")"
             << " to match type of corresponding result (" << std::get<1>(it)
             << ")";
  }

  // Output tensor indexing map may not depend on reduction indices.
  for (OpOperand &opOperand : linalgOp.getOutputOpOperands()) {
    AffineMap outputMap = linalgOp.getIndexingMap(opOperand.getOperandNumber());
    for (auto expr : outputMap.getResults()) {
      for (auto dim : redDims) {
        unsigned pos = dim.cast<AffineDimExpr>().getPosition();
        if (expr.isFunctionOfDim(pos)) {
          std::string exprStr;
          {
            llvm::raw_string_ostream os(exprStr);
            os << expr;
          }
          return op->emitError(
                     "unexpected output tensor expression in indexing map #")
                 << (opOperand.getOperandNumber() - linalgOp.getNumInputs())
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

  if (linalgOp.getNumShapedOperands() + numBBIvs != block.getNumArguments())
    return op->emitError("expected as many non-induction variable region "
                         "arguments as the number of shaped operands");

  // Note: the number and type of yield values are checked in the YieldOp.
  for (unsigned i = 0; i < numBBIvs; ++i)
    if (!block.getArgument(i).getType().isIndex())
      return op->emitOpError("expected index block argument #") << i;

  unsigned idx = 0;
  for (auto it : llvm::zip(linalgOp.getShapedOperandTypes(),
                           block.getArguments().drop_front(numBBIvs))) {
    if (std::get<0>(it).getElementType() != std::get<1>(it).getType())
      return op->emitError("expected type of bb argument #")
             << (idx + numBBIvs) << " (" << std::get<1>(it).getType() << ")"
             << " to match element type of corresponding shaped operand ("
             << std::get<0>(it).getElementType() << ")";
    ++idx;
  }

  return success();
}
