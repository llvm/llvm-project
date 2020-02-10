//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::edsc;

static Optional<ValueHandle> emitStaticFor(ArrayRef<ValueHandle> lbs,
                                           ArrayRef<ValueHandle> ubs,
                                           int64_t step) {
  if (lbs.size() != 1 || ubs.size() != 1)
    return Optional<ValueHandle>();

  auto *lbDef = lbs.front().getValue().getDefiningOp();
  auto *ubDef = ubs.front().getValue().getDefiningOp();
  if (!lbDef || !ubDef)
    return Optional<ValueHandle>();

  auto lbConst = dyn_cast<ConstantIndexOp>(lbDef);
  auto ubConst = dyn_cast<ConstantIndexOp>(ubDef);
  if (!lbConst || !ubConst)
    return Optional<ValueHandle>();

  return ValueHandle(ScopedContext::getBuilder()
                         .create<AffineForOp>(ScopedContext::getLocation(),
                                              lbConst.getValue(),
                                              ubConst.getValue(), step)
                         .getInductionVar());
}

LoopBuilder mlir::edsc::makeAffineLoopBuilder(ValueHandle *iv,
                                              ArrayRef<ValueHandle> lbHandles,
                                              ArrayRef<ValueHandle> ubHandles,
                                              int64_t step) {
  mlir::edsc::LoopBuilder result;
  if (auto staticFor = emitStaticFor(lbHandles, ubHandles, step)) {
    *iv = staticFor.getValue();
  } else {
    SmallVector<Value, 4> lbs(lbHandles.begin(), lbHandles.end());
    SmallVector<Value, 4> ubs(ubHandles.begin(), ubHandles.end());
    auto b = ScopedContext::getBuilder();
    *iv = ValueHandle(
        b.create<AffineForOp>(ScopedContext::getLocation(), lbs,
                              b.getMultiDimIdentityMap(lbs.size()), ubs,
                              b.getMultiDimIdentityMap(ubs.size()), step)
            .getInductionVar());
  }
  auto *body = getForInductionVarOwner(iv->getValue()).getBody();
  result.enter(body, /*prev=*/1);
  return result;
}

mlir::edsc::AffineLoopNestBuilder::AffineLoopNestBuilder(
    ValueHandle *iv, ArrayRef<ValueHandle> lbs, ArrayRef<ValueHandle> ubs,
    int64_t step) {
  loops.emplace_back(makeAffineLoopBuilder(iv, lbs, ubs, step));
}

mlir::edsc::AffineLoopNestBuilder::AffineLoopNestBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> lbs,
    ArrayRef<ValueHandle> ubs, ArrayRef<int64_t> steps) {
  assert(ivs.size() == lbs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == steps.size() && "Mismatch in number of arguments");
  for (auto it : llvm::zip(ivs, lbs, ubs, steps))
    loops.emplace_back(makeAffineLoopBuilder(std::get<0>(it), std::get<1>(it),
                                             std::get<2>(it), std::get<3>(it)));
}

void mlir::edsc::AffineLoopNestBuilder::operator()(
    function_ref<void(void)> fun) {
  if (fun)
    fun();
  // Iterate on the calling operator() on all the loops in the nest.
  // The iteration order is from innermost to outermost because enter/exit needs
  // to be asymmetric (i.e. enter() occurs on LoopBuilder construction, exit()
  // occurs on calling operator()). The asymmetry is required for properly
  // nesting imperfectly nested regions (see LoopBuilder::operator()).
  for (auto lit = loops.rbegin(), eit = loops.rend(); lit != eit; ++lit)
    (*lit)();
}

template <typename Op>
static ValueHandle createBinaryHandle(ValueHandle lhs, ValueHandle rhs) {
  return ValueHandle::create<Op>(lhs.getValue(), rhs.getValue());
}

static std::pair<AffineExpr, Value>
categorizeValueByAffineType(MLIRContext *context, Value val, unsigned &numDims,
                            unsigned &numSymbols) {
  AffineExpr d;
  Value resultVal = nullptr;
  if (auto constant = dyn_cast_or_null<ConstantIndexOp>(val.getDefiningOp())) {
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

static ValueHandle createBinaryIndexHandle(
    ValueHandle lhs, ValueHandle rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  MLIRContext *context = ScopedContext::getContext();
  unsigned numDims = 0, numSymbols = 0;
  AffineExpr d0, d1;
  Value v0, v1;
  std::tie(d0, v0) =
      categorizeValueByAffineType(context, lhs.getValue(), numDims, numSymbols);
  std::tie(d1, v1) =
      categorizeValueByAffineType(context, rhs.getValue(), numDims, numSymbols);
  SmallVector<Value, 2> operands;
  if (v0) {
    operands.push_back(v0);
  }
  if (v1) {
    operands.push_back(v1);
  }
  auto map = AffineMap::get(numDims, numSymbols, {affCombiner(d0, d1)});
  // TODO: createOrFold when available.
  Operation *op =
      makeComposedAffineApply(ScopedContext::getBuilder(),
                              ScopedContext::getLocation(), map, operands)
          .getOperation();
  assert(op->getNumResults() == 1 && "Expected single result AffineApply");
  return ValueHandle(op->getResult(0));
}

template <typename IOp, typename FOp>
static ValueHandle createBinaryHandle(
    ValueHandle lhs, ValueHandle rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  auto thisType = lhs.getValue().getType();
  auto thatType = rhs.getValue().getType();
  assert(thisType == thatType && "cannot mix types in operators");
  (void)thisType;
  (void)thatType;
  if (thisType.isIndex()) {
    return createBinaryIndexHandle(lhs, rhs, affCombiner);
  } else if (thisType.isa<IntegerType>()) {
    return createBinaryHandle<IOp>(lhs, rhs);
  } else if (thisType.isa<FloatType>()) {
    return createBinaryHandle<FOp>(lhs, rhs);
  } else if (thisType.isa<VectorType>() || thisType.isa<TensorType>()) {
    auto aggregateType = thisType.cast<ShapedType>();
    if (aggregateType.getElementType().isa<IntegerType>())
      return createBinaryHandle<IOp>(lhs, rhs);
    else if (aggregateType.getElementType().isa<FloatType>())
      return createBinaryHandle<FOp>(lhs, rhs);
  }
  llvm_unreachable("failed to create a ValueHandle");
}

ValueHandle mlir::edsc::op::operator+(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<AddIOp, AddFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 + d1; });
}

ValueHandle mlir::edsc::op::operator-(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<SubIOp, SubFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 - d1; });
}

ValueHandle mlir::edsc::op::operator*(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<MulIOp, MulFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 * d1; });
}

ValueHandle mlir::edsc::op::operator/(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<SignedDivIOp, DivFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) -> AffineExpr {
        llvm_unreachable("only exprs of non-index type support operator/");
      });
}

ValueHandle mlir::edsc::op::operator%(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<SignedRemIOp, RemFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 % d1; });
}

ValueHandle mlir::edsc::op::floorDiv(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.floorDiv(d1); });
}

ValueHandle mlir::edsc::op::ceilDiv(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.ceilDiv(d1); });
}

ValueHandle mlir::edsc::op::operator!(ValueHandle value) {
  assert(value.getType().isInteger(1) && "expected boolean expression");
  return ValueHandle::create<ConstantIntOp>(1, 1) - value;
}

ValueHandle mlir::edsc::op::operator&&(ValueHandle lhs, ValueHandle rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return lhs * rhs;
}

ValueHandle mlir::edsc::op::operator||(ValueHandle lhs, ValueHandle rhs) {
  return !(!lhs && !rhs);
}

static ValueHandle createIComparisonExpr(CmpIPredicate predicate,
                                         ValueHandle lhs, ValueHandle rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert((lhsType.isa<IndexType>() || lhsType.isa<IntegerType>()) &&
         "only integer comparisons are supported");

  auto op = ScopedContext::getBuilder().create<CmpIOp>(
      ScopedContext::getLocation(), predicate, lhs.getValue(), rhs.getValue());
  return ValueHandle(op.getResult());
}

static ValueHandle createFComparisonExpr(CmpFPredicate predicate,
                                         ValueHandle lhs, ValueHandle rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert(lhsType.isa<FloatType>() && "only float comparisons are supported");

  auto op = ScopedContext::getBuilder().create<CmpFOp>(
      ScopedContext::getLocation(), predicate, lhs.getValue(), rhs.getValue());
  return ValueHandle(op.getResult());
}

// All floating point comparison are ordered through EDSL
ValueHandle mlir::edsc::op::operator==(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OEQ, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::eq, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator!=(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::ONE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ne, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator<(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             :
             // TODO(ntv,zinenko): signed by default, how about unsigned?
             createIComparisonExpr(CmpIPredicate::slt, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator<=(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sle, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator>(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sgt, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator>=(ValueHandle lhs, ValueHandle rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sge, lhs, rhs);
}
