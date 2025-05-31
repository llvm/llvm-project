//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::affine;

namespace mlir {
namespace {

struct AffineApplyOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineApplyOpInterface,
                                                   AffineApplyOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto applyOp = cast<AffineApplyOp>(op);
    assert(value == applyOp.getResult() && "invalid value");
    assert(applyOp.getAffineMap().getNumResults() == 1 &&
           "expected single result");

    // Fully compose this affine.apply with other ops because the folding logic
    // can see opportunities for simplifying the affine map that
    // `FlatLinearConstraints` can currently not see.
    AffineMap map = applyOp.getAffineMap();
    SmallVector<Value> operands = llvm::to_vector(applyOp.getOperands());
    fullyComposeAffineMapAndOperands(&map, &operands);

    // Align affine map result with dims/symbols in the constraint set.
    AffineExpr expr = map.getResult(0);
    SmallVector<AffineExpr> dimReplacements, symReplacements;
    for (int64_t i = 0, e = map.getNumDims(); i < e; ++i)
      dimReplacements.push_back(cstr.getExpr(operands[i]));
    for (int64_t i = map.getNumDims(),
                 e = map.getNumDims() + map.getNumSymbols();
         i < e; ++i)
      symReplacements.push_back(cstr.getExpr(operands[i]));
    AffineExpr bound =
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
    cstr.bound(value) == bound;
  }
};

struct AffineMinOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMinOpInterface,
                                                   AffineMinOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<AffineMinOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : minOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          minOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          minOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      cstr.bound(value) <= bound;
    }
  };
};

struct AffineMaxOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMaxOpInterface,
                                                   AffineMaxOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<AffineMaxOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : maxOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      cstr.bound(value) >= bound;
    }
  };
};

struct AffineDelinearizeIndexOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          AffineDelinearizeIndexOpInterface, AffineDelinearizeIndexOp> {
  void populateBoundsForIndexValue(Operation *rawOp, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto op = cast<AffineDelinearizeIndexOp>(rawOp);
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == rawOp &&
           "bounded value isn't a result of this delinearize_index");
    unsigned resIdx = result.getResultNumber();

    AffineExpr linearIdx = cstr.getExpr(op.getLinearIndex());

    SmallVector<OpFoldResult> basis = op.getPaddedBasis();
    AffineExpr divisor = cstr.getExpr(1);
    for (OpFoldResult basisElem : llvm::drop_begin(basis, resIdx + 1))
      divisor = divisor * cstr.getExpr(basisElem);

    if (resIdx == 0) {
      cstr.bound(value) == linearIdx.floorDiv(divisor);
      if (!basis.front().isNull())
        cstr.bound(value) < cstr.getExpr(basis.front());
      return;
    }
    AffineExpr thisBasis = cstr.getExpr(basis[resIdx]);
    cstr.bound(value) == (linearIdx % (thisBasis * divisor)).floorDiv(divisor);
  }
};

struct AffineLinearizeIndexOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          AffineLinearizeIndexOpInterface, AffineLinearizeIndexOp> {
  void populateBoundsForIndexValue(Operation *rawOp, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto op = cast<AffineLinearizeIndexOp>(rawOp);
    assert(value == op.getResult() &&
           "value isn't the result of this linearize");

    AffineExpr bound = cstr.getExpr(0);
    AffineExpr stride = cstr.getExpr(1);
    SmallVector<OpFoldResult> basis = op.getPaddedBasis();
    OperandRange multiIndex = op.getMultiIndex();
    unsigned numArgs = multiIndex.size();
    for (auto [revArgNum, length] : llvm::enumerate(llvm::reverse(basis))) {
      unsigned argNum = numArgs - (revArgNum + 1);
      if (argNum == 0)
        break;
      OpFoldResult indexAsFoldRes = getAsOpFoldResult(multiIndex[argNum]);
      bound = bound + cstr.getExpr(indexAsFoldRes) * stride;
      stride = stride * cstr.getExpr(length);
    }
    bound = bound + cstr.getExpr(op.getMultiIndex().front()) * stride;
    cstr.bound(value) == bound;
    if (op.getDisjoint() && !basis.front().isNull()) {
      cstr.bound(value) < stride *cstr.getExpr(basis.front());
    }
  }
};
} // namespace
} // namespace mlir

void mlir::affine::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AffineDialect *dialect) {
    AffineApplyOp::attachInterface<AffineApplyOpInterface>(*ctx);
    AffineMaxOp::attachInterface<AffineMaxOpInterface>(*ctx);
    AffineMinOp::attachInterface<AffineMinOpInterface>(*ctx);
    AffineDelinearizeIndexOp::attachInterface<
        AffineDelinearizeIndexOpInterface>(*ctx);
    AffineLinearizeIndexOp::attachInterface<AffineLinearizeIndexOpInterface>(
        *ctx);
  });
}

FailureOr<int64_t>
mlir::affine::fullyComposeAndComputeConstantDelta(Value value1, Value value2) {
  assert(value1.getType().isIndex() && "expected index type");
  assert(value2.getType().isIndex() && "expected index type");

  // Subtract the two values/dimensions from each other. If the result is 0,
  // both are equal.
  Builder b(value1.getContext());
  AffineMap map = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                 b.getAffineDimExpr(0) - b.getAffineDimExpr(1));
  // Fully compose the affine map with other ops because the folding logic
  // can see opportunities for simplifying the affine map that
  // `FlatLinearConstraints` can currently not see.
  SmallVector<Value> mapOperands;
  mapOperands.push_back(value1);
  mapOperands.push_back(value2);
  affine::fullyComposeAffineMapAndOperands(&map, &mapOperands);
  return ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::EQ,
      ValueBoundsConstraintSet::Variable(map, mapOperands));
}
