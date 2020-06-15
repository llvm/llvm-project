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

static Optional<Value> emitStaticFor(ArrayRef<Value> lbs, ArrayRef<Value> ubs,
                                     int64_t step) {
  if (lbs.size() != 1 || ubs.size() != 1)
    return Optional<Value>();

  auto *lbDef = lbs.front().getDefiningOp();
  auto *ubDef = ubs.front().getDefiningOp();
  if (!lbDef || !ubDef)
    return Optional<Value>();

  auto lbConst = dyn_cast<ConstantIndexOp>(lbDef);
  auto ubConst = dyn_cast<ConstantIndexOp>(ubDef);
  if (!lbConst || !ubConst)
    return Optional<Value>();
  return ScopedContext::getBuilderRef()
      .create<AffineForOp>(ScopedContext::getLocation(), lbConst.getValue(),
                           ubConst.getValue(), step)
      .getInductionVar();
}

LoopBuilder mlir::edsc::makeAffineLoopBuilder(Value *iv, ArrayRef<Value> lbs,
                                              ArrayRef<Value> ubs,
                                              int64_t step) {
  mlir::edsc::LoopBuilder result;
  if (auto staticForIv = emitStaticFor(lbs, ubs, step))
    *iv = staticForIv.getValue();
  else
    *iv = ScopedContext::getBuilderRef()
              .create<AffineForOp>(
                  ScopedContext::getLocation(), lbs,
                  ScopedContext::getBuilderRef().getMultiDimIdentityMap(
                      lbs.size()),
                  ubs,
                  ScopedContext::getBuilderRef().getMultiDimIdentityMap(
                      ubs.size()),
                  step)
              .getInductionVar();

  auto *body = getForInductionVarOwner(*iv).getBody();
  result.enter(body);
  return result;
}

mlir::edsc::AffineLoopNestBuilder::AffineLoopNestBuilder(Value *iv,
                                                         ArrayRef<Value> lbs,
                                                         ArrayRef<Value> ubs,
                                                         int64_t step) {
  loops.emplace_back(makeAffineLoopBuilder(iv, lbs, ubs, step));
}

void mlir::edsc::affineLoopNestBuilder(
    ValueRange lbs, ValueRange ubs, ArrayRef<int64_t> steps,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lbs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(lbs.size() == steps.size() && "Mismatch in number of arguments");

  // If there are no loops to be constructed, construct the body anyway.
  if (lbs.empty()) {
    if (bodyBuilderFn)
      bodyBuilderFn(ValueRange());
    return;
  }

  // Fetch the builder and location.
  OpBuilder &builder = ScopedContext::getBuilderRef();
  OpBuilder::InsertionGuard guard(builder);
  Location loc = ScopedContext::getLocation();
  AffineMap identity = builder.getDimIdentityMap();

  // Create the loops iteratively and store the induction variables.
  SmallVector<Value, 4> ivs;
  ivs.reserve(lbs.size());
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    // Callback for creating the loop body, always creates the terminator.
    auto loopBody = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                        Value iv) {
      ivs.push_back(iv);
      // In the innermost loop, call the body builder.
      if (i == e - 1 && bodyBuilderFn) {
        ScopedContext nestedContext(nestedBuilder, loc);
        OpBuilder::InsertionGuard nestedGuard(nestedBuilder);
        bodyBuilderFn(ivs);
      }
      nestedBuilder.create<AffineTerminatorOp>(nestedLoc);
    };

    // Create the loop. If the bounds are known to be constants, use the
    // constant form of the loop.
    auto lbConst = lbs[i].getDefiningOp<ConstantIndexOp>();
    auto ubConst = ubs[i].getDefiningOp<ConstantIndexOp>();
    auto loop = lbConst && ubConst
                    ? builder.create<AffineForOp>(loc, lbConst.getValue(),
                                                  ubConst.getValue(), steps[i],
                                                  loopBody)
                    : builder.create<AffineForOp>(loc, lbs[i], identity, ubs[i],
                                                  identity, steps[i], loopBody);
    builder.setInsertionPointToStart(loop.getBody());
  }
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
      builder.getMultiDimIdentityMap(ubs.size()), step,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
        }
        nestedBuilder.create<AffineTerminatorOp>(nestedLoc);
      });
}

mlir::edsc::AffineLoopNestBuilder::AffineLoopNestBuilder(
    MutableArrayRef<Value> ivs, ArrayRef<Value> lbs, ArrayRef<Value> ubs,
    ArrayRef<int64_t> steps) {
  assert(ivs.size() == lbs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == steps.size() && "Mismatch in number of arguments");
  for (auto it : llvm::zip(ivs, lbs, ubs, steps))
    loops.emplace_back(makeAffineLoopBuilder(&std::get<0>(it), std::get<1>(it),
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
  } else if (thisType.isa<VectorType>() || thisType.isa<TensorType>()) {
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
Value mlir::edsc::op::operator<(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             :
             // TODO(ntv,zinenko): signed by default, how about unsigned?
             createIComparisonExpr(CmpIPredicate::slt, lhs, rhs);
}
Value mlir::edsc::op::operator<=(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sle, lhs, rhs);
}
Value mlir::edsc::op::operator>(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sgt, lhs, rhs);
}
Value mlir::edsc::op::operator>=(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sge, lhs, rhs);
}
