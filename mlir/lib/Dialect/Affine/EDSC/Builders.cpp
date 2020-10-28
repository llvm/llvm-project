//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::edsc;

void mlir::edsc::affineLoopNestBuilder(
    ValueRange lbs, ValueRange ubs, ArrayRef<int64_t> steps,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");

  // Wrap the body builder function into an interface compatible with the main
  // builder.
  auto wrappedBuilderFn = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                              ValueRange ivs) {
    ScopedContext context(nestedBuilder, nestedLoc);
    bodyBuilderFn(ivs);
  };
  function_ref<void(OpBuilder &, Location, ValueRange)> wrapper;
  if (bodyBuilderFn)
    wrapper = wrappedBuilderFn;

  // Extract the builder, location and construct the loop nest.
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  buildAffineLoopNest(builder, loc, lbs, ubs, steps, wrapper);
}

void mlir::edsc::affineLoopBuilder(ValueRange lbs, ValueRange ubs, int64_t step,
                                   function_ref<void(Value)> bodyBuilderFn) {
  // Fetch the builder and location.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();

  // Create the actual loop and call the body builder, if provided, after
  // updating the scoped context.
  builder.create<AffineForOp>(
      loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
      builder.getMultiDimIdentityMap(ubs.size()), step, llvm::None,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
        }
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });
}

void mlir::edsc::affineLoopBuilder(
    ValueRange lbs, ValueRange ubs, int64_t step, ValueRange iterArgs,
    function_ref<void(Value, ValueRange)> bodyBuilderFn) {
  // Fetch the builder and location.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();

  // Create the actual loop and call the body builder, if provided, after
  // updating the scoped context.
  builder.create<AffineForOp>(
      loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
      builder.getMultiDimIdentityMap(ubs.size()), step, iterArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv, itrArgs);
        } else if (itrArgs.empty())
          nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });
}

static std::pair<AffineExpr, Value>
categorizeValueByAffineType(MLIRContext *context, Value val, unsigned &numDims,
                            unsigned &numSymbols) {
  AffineExpr d;
  Value resultVal = nullptr;
  if (auto constant = val.getDefiningOp<ConstantIndexOp>()) {
    d = getAffineConstantExpr(constant.getValue(), context);
  } else if (isValidSymbol(val) && !isValidDim(val)) {
    d = getAffineSymbolExpr(numSymbols++, context);
    resultVal = val;
  } else {
    d = getAffineDimExpr(numDims++, context);
    resultVal = val;
  }
  return std::make_pair(d, resultVal);
}

static Value createBinaryIndexHandle(
    Value lhs, Value rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  MLIRContext *context = ScopedContext::getContext();
  unsigned numDims = 0, numSymbols = 0;
  AffineExpr d0, d1;
  Value v0, v1;
  std::tie(d0, v0) =
      categorizeValueByAffineType(context, lhs, numDims, numSymbols);
  std::tie(d1, v1) =
      categorizeValueByAffineType(context, rhs, numDims, numSymbols);
  SmallVector<Value, 2> operands;
  if (v0)
    operands.push_back(v0);
  if (v1)
    operands.push_back(v1);
  auto map = AffineMap::get(numDims, numSymbols, affCombiner(d0, d1));

  // TODO: createOrFold when available.
  Operation *op =
      makeComposedAffineApply(ScopedContext::getBuilderRef(),
                              ScopedContext::getLocation(), map, operands)
          .getOperation();
  assert(op->getNumResults() == 1 && "Expected single result AffineApply");
  return op->getResult(0);
}

template <typename IOp, typename FOp>
static Value createBinaryHandle(
    Value lhs, Value rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  auto thisType = lhs.getType();
  auto thatType = rhs.getType();
  assert(thisType == thatType && "cannot mix types in operators");
  (void)thisType;
  (void)thatType;
  if (thisType.isIndex()) {
    return createBinaryIndexHandle(lhs, rhs, affCombiner);
  } else if (thisType.isSignlessInteger()) {
    return ValueBuilder<IOp>(lhs, rhs);
  } else if (thisType.isa<FloatType>()) {
    return ValueBuilder<FOp>(lhs, rhs);
  } else if (thisType.isa<VectorType, TensorType>()) {
    auto aggregateType = thisType.cast<ShapedType>();
    if (aggregateType.getElementType().isSignlessInteger())
      return ValueBuilder<IOp>(lhs, rhs);
    else if (aggregateType.getElementType().isa<FloatType>())
      return ValueBuilder<FOp>(lhs, rhs);
  }
  llvm_unreachable("failed to create a Value");
}

Value mlir::edsc::op::operator+(Value lhs, Value rhs) {
  return createBinaryHandle<AddIOp, AddFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 + d1; });
}

Value mlir::edsc::op::operator-(Value lhs, Value rhs) {
  return createBinaryHandle<SubIOp, SubFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 - d1; });
}

Value mlir::edsc::op::operator*(Value lhs, Value rhs) {
  return createBinaryHandle<MulIOp, MulFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 * d1; });
}

Value mlir::edsc::op::operator/(Value lhs, Value rhs) {
  return createBinaryHandle<SignedDivIOp, DivFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) -> AffineExpr {
        llvm_unreachable("only exprs of non-index type support operator/");
      });
}

Value mlir::edsc::op::operator%(Value lhs, Value rhs) {
  return createBinaryHandle<SignedRemIOp, RemFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 % d1; });
}

Value mlir::edsc::op::floorDiv(Value lhs, Value rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.floorDiv(d1); });
}

Value mlir::edsc::op::ceilDiv(Value lhs, Value rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.ceilDiv(d1); });
}

Value mlir::edsc::op::negate(Value value) {
  assert(value.getType().isInteger(1) && "expected boolean expression");
  return ValueBuilder<ConstantIntOp>(1, 1) - value;
}

Value mlir::edsc::op::operator&&(Value lhs, Value rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return ValueBuilder<AndOp>(lhs, rhs);
}

Value mlir::edsc::op::operator||(Value lhs, Value rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return ValueBuilder<OrOp>(lhs, rhs);
}

static Value createIComparisonExpr(CmpIPredicate predicate, Value lhs,
                                   Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert((lhsType.isa<IndexType>() || lhsType.isSignlessInteger()) &&
         "only integer comparisons are supported");

  return ScopedContext::getBuilderRef().create<CmpIOp>(
      ScopedContext::getLocation(), predicate, lhs, rhs);
}

static Value createFComparisonExpr(CmpFPredicate predicate, Value lhs,
                                   Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert(lhsType.isa<FloatType>() && "only float comparisons are supported");

  return ScopedContext::getBuilderRef().create<CmpFOp>(
      ScopedContext::getLocation(), predicate, lhs, rhs);
}

// All floating point comparison are ordered through EDSL
Value mlir::edsc::op::eq(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OEQ, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::eq, lhs, rhs);
}
Value mlir::edsc::op::ne(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::ONE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ne, lhs, rhs);
}
Value mlir::edsc::op::slt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::slt, lhs, rhs);
}
Value mlir::edsc::op::sle(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sle, lhs, rhs);
}
Value mlir::edsc::op::sgt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sgt, lhs, rhs);
}
Value mlir::edsc::op::sge(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sge, lhs, rhs);
}
Value mlir::edsc::op::ult(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ult, lhs, rhs);
}
Value mlir::edsc::op::ule(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ule, lhs, rhs);
}
Value mlir::edsc::op::ugt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ugt, lhs, rhs);
}
Value mlir::edsc::op::uge(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::uge, lhs, rhs);
}
