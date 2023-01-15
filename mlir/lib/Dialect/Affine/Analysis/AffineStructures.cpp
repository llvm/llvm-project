//===- AffineStructures.cpp - MLIR Affine Structures Class-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for affine/polyhedral analysis of affine dialect ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "affine-structures"

using namespace mlir;
using namespace presburger;

namespace {

// See comments for SimpleAffineExprFlattener.
// An AffineExprFlattener extends a SimpleAffineExprFlattener by recording
// constraint information associated with mod's, floordiv's, and ceildiv's
// in FlatAffineValueConstraints 'localVarCst'.
struct AffineExprFlattener : public SimpleAffineExprFlattener {
public:
  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  IntegerPolyhedron localVarCst;

  AffineExprFlattener(unsigned nDims, unsigned nSymbols)
      : SimpleAffineExprFlattener(nDims, nSymbols),
        localVarCst(PresburgerSpace::getSetSpace(nDims, nSymbols)) {}

private:
  // Add a local variable (needed to flatten a mod, floordiv, ceildiv expr).
  // The local variable added is always a floordiv of a pure add/mul affine
  // function of other variables, coefficients of which are specified in
  // `dividend' and with respect to the positive constant `divisor'. localExpr
  // is the simplified tree expression (AffineExpr) corresponding to the
  // quantifier.
  void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                          AffineExpr localExpr) override {
    SimpleAffineExprFlattener::addLocalFloorDivId(dividend, divisor, localExpr);
    // Update localVarCst.
    localVarCst.addLocalFloorDiv(dividend, divisor);
  }
};

} // namespace

// Flattens the expressions in map. Returns failure if 'expr' was unable to be
// flattened (i.e., semi-affine expressions not handled yet).
static LogicalResult
getFlattenedAffineExprs(ArrayRef<AffineExpr> exprs, unsigned numDims,
                        unsigned numSymbols,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatAffineValueConstraints *localVarCst) {
  if (exprs.empty()) {
    localVarCst->reset(numDims, numSymbols);
    return success();
  }

  AffineExprFlattener flattener(numDims, numSymbols);
  // Use the same flattener to simplify each expression successively. This way
  // local variables / expressions are shared.
  for (auto expr : exprs) {
    if (!expr.isPureAffine())
      return failure();

    flattener.walkPostOrder(expr);
  }

  assert(flattener.operandExprStack.size() == exprs.size());
  flattenedExprs->clear();
  flattenedExprs->assign(flattener.operandExprStack.begin(),
                         flattener.operandExprStack.end());

  if (localVarCst)
    localVarCst->clearAndCopyFrom(flattener.localVarCst);

  return success();
}

// Flattens 'expr' into 'flattenedExpr'. Returns failure if 'expr' was unable to
// be flattened (semi-affine expressions not handled yet).
LogicalResult
mlir::getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                             unsigned numSymbols,
                             SmallVectorImpl<int64_t> *flattenedExpr,
                             FlatAffineValueConstraints *localVarCst) {
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  LogicalResult ret = ::getFlattenedAffineExprs({expr}, numDims, numSymbols,
                                                &flattenedExprs, localVarCst);
  *flattenedExpr = flattenedExprs[0];
  return ret;
}

/// Flattens the expressions in map. Returns failure if 'expr' was unable to be
/// flattened (i.e., semi-affine expressions not handled yet).
LogicalResult mlir::getFlattenedAffineExprs(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineValueConstraints *localVarCst) {
  if (map.getNumResults() == 0) {
    localVarCst->reset(map.getNumDims(), map.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(map.getResults(), map.getNumDims(),
                                   map.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

LogicalResult mlir::getFlattenedAffineExprs(
    IntegerSet set, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineValueConstraints *localVarCst) {
  if (set.getNumConstraints() == 0) {
    localVarCst->reset(set.getNumDims(), set.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(set.getConstraints(), set.getNumDims(),
                                   set.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

//===----------------------------------------------------------------------===//
// FlatAffineConstraints / FlatAffineValueConstraints.
//===----------------------------------------------------------------------===//

std::unique_ptr<FlatAffineValueConstraints>
FlatAffineValueConstraints::clone() const {
  return std::make_unique<FlatAffineValueConstraints>(*this);
}

// Construct from an IntegerSet.
FlatAffineValueConstraints::FlatAffineValueConstraints(IntegerSet set,
                                                       ValueRange operands)
    : IntegerPolyhedron(set.getNumInequalities(), set.getNumEqualities(),
                        set.getNumDims() + set.getNumSymbols() + 1,
                        PresburgerSpace::getSetSpace(set.getNumDims(),
                                                     set.getNumSymbols(),
                                                     /*numLocals=*/0)) {
  // Populate values.
  if (operands.empty()) {
    values.resize(getNumDimAndSymbolVars(), std::nullopt);
  } else {
    assert(set.getNumInputs() == operands.size() && "operand count mismatch");
    values.assign(operands.begin(), operands.end());
  }

  // Flatten expressions and add them to the constraint system.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineValueConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(set, &flatExprs, &localVarCst))) {
    assert(false && "flattening unimplemented for semi-affine integer sets");
    return;
  }
  assert(flatExprs.size() == set.getNumConstraints());
  insertVar(VarKind::Local, getNumVarKind(VarKind::Local),
            /*num=*/localVarCst.getNumLocalVars());

  for (unsigned i = 0, e = flatExprs.size(); i < e; ++i) {
    const auto &flatExpr = flatExprs[i];
    assert(flatExpr.size() == getNumCols());
    if (set.getEqFlags()[i]) {
      addEquality(flatExpr);
    } else {
      addInequality(flatExpr);
    }
  }
  // Add the other constraints involving local vars from flattening.
  append(localVarCst);
}

// Construct a hyperrectangular constraint set from ValueRanges that represent
// induction variables, lower and upper bounds. `ivs`, `lbs` and `ubs` are
// expected to match one to one. The order of variables and constraints is:
//
// ivs | lbs | ubs | eq/ineq
// ----+-----+-----+---------
//   1   -1     0      >= 0
// ----+-----+-----+---------
//  -1    0     1      >= 0
//
// All dimensions as set as VarKind::SetDim.
FlatAffineValueConstraints
FlatAffineValueConstraints::getHyperrectangular(ValueRange ivs, ValueRange lbs,
                                                ValueRange ubs) {
  FlatAffineValueConstraints res;
  unsigned nIvs = ivs.size();
  assert(nIvs == lbs.size() && "expected as many lower bounds as ivs");
  assert(nIvs == ubs.size() && "expected as many upper bounds as ivs");

  if (nIvs == 0)
    return res;

  res.appendDimVar(ivs);
  unsigned lbsStart = res.appendDimVar(lbs);
  unsigned ubsStart = res.appendDimVar(ubs);

  MLIRContext *ctx = ivs.front().getContext();
  for (int ivIdx = 0, e = nIvs; ivIdx < e; ++ivIdx) {
    // iv - lb >= 0
    AffineMap lb = AffineMap::get(/*dimCount=*/3 * nIvs, /*symbolCount=*/0,
                                  getAffineDimExpr(lbsStart + ivIdx, ctx));
    if (failed(res.addBound(BoundType::LB, ivIdx, lb)))
      llvm_unreachable("Unexpected FlatAffineValueConstraints creation error");
    // -iv + ub >= 0
    AffineMap ub = AffineMap::get(/*dimCount=*/3 * nIvs, /*symbolCount=*/0,
                                  getAffineDimExpr(ubsStart + ivIdx, ctx));
    if (failed(res.addBound(BoundType::UB, ivIdx, ub)))
      llvm_unreachable("Unexpected FlatAffineValueConstraints creation error");
  }
  return res;
}

void FlatAffineValueConstraints::reset(unsigned numReservedInequalities,
                                       unsigned numReservedEqualities,
                                       unsigned newNumReservedCols,
                                       unsigned newNumDims,
                                       unsigned newNumSymbols,
                                       unsigned newNumLocals) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  *this = FlatAffineValueConstraints(numReservedInequalities,
                                     numReservedEqualities, newNumReservedCols,
                                     newNumDims, newNumSymbols, newNumLocals);
}

void FlatAffineValueConstraints::reset(unsigned newNumDims,
                                       unsigned newNumSymbols,
                                       unsigned newNumLocals) {
  reset(/*numReservedInequalities=*/0, /*numReservedEqualities=*/0,
        /*numReservedCols=*/newNumDims + newNumSymbols + newNumLocals + 1,
        newNumDims, newNumSymbols, newNumLocals);
}

void FlatAffineValueConstraints::reset(
    unsigned numReservedInequalities, unsigned numReservedEqualities,
    unsigned newNumReservedCols, unsigned newNumDims, unsigned newNumSymbols,
    unsigned newNumLocals, ArrayRef<Value> valArgs) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  SmallVector<Optional<Value>, 8> newVals;
  if (!valArgs.empty())
    newVals.assign(valArgs.begin(), valArgs.end());

  *this = FlatAffineValueConstraints(
      numReservedInequalities, numReservedEqualities, newNumReservedCols,
      newNumDims, newNumSymbols, newNumLocals, newVals);
}

void FlatAffineValueConstraints::reset(unsigned newNumDims,
                                       unsigned newNumSymbols,
                                       unsigned newNumLocals,
                                       ArrayRef<Value> valArgs) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals, valArgs);
}

unsigned FlatAffineValueConstraints::appendDimVar(ValueRange vals) {
  unsigned pos = getNumDimVars();
  return insertVar(VarKind::SetDim, pos, vals);
}

unsigned FlatAffineValueConstraints::appendSymbolVar(ValueRange vals) {
  unsigned pos = getNumSymbolVars();
  return insertVar(VarKind::Symbol, pos, vals);
}

unsigned FlatAffineValueConstraints::insertDimVar(unsigned pos,
                                                  ValueRange vals) {
  return insertVar(VarKind::SetDim, pos, vals);
}

unsigned FlatAffineValueConstraints::insertSymbolVar(unsigned pos,
                                                     ValueRange vals) {
  return insertVar(VarKind::Symbol, pos, vals);
}

unsigned FlatAffineValueConstraints::insertVar(VarKind kind, unsigned pos,
                                               unsigned num) {
  unsigned absolutePos = IntegerPolyhedron::insertVar(kind, pos, num);

  if (kind != VarKind::Local) {
    values.insert(values.begin() + absolutePos, num, std::nullopt);
    assert(values.size() == getNumDimAndSymbolVars());
  }

  return absolutePos;
}

unsigned FlatAffineValueConstraints::insertVar(VarKind kind, unsigned pos,
                                               ValueRange vals) {
  assert(!vals.empty() && "expected ValueRange with Values.");
  assert(kind != VarKind::Local &&
         "values cannot be attached to local variables.");
  unsigned num = vals.size();
  unsigned absolutePos = IntegerPolyhedron::insertVar(kind, pos, num);

  // If a Value is provided, insert it; otherwise use None.
  for (unsigned i = 0; i < num; ++i)
    values.insert(values.begin() + absolutePos + i,
                  vals[i] ? Optional<Value>(vals[i]) : std::nullopt);

  assert(values.size() == getNumDimAndSymbolVars());
  return absolutePos;
}

bool FlatAffineValueConstraints::hasValues() const {
  return llvm::any_of(
      values, [](const Optional<Value> &var) { return var.has_value(); });
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of variables, appearing in the same order.
static bool areVarsAligned(const FlatAffineValueConstraints &a,
                           const FlatAffineValueConstraints &b) {
  return a.getNumDimVars() == b.getNumDimVars() &&
         a.getNumSymbolVars() == b.getNumSymbolVars() &&
         a.getNumVars() == b.getNumVars() &&
         a.getMaybeValues().equals(b.getMaybeValues());
}

/// Calls areVarsAligned to check if two constraint systems have the same set
/// of variables in the same order.
bool FlatAffineValueConstraints::areVarsAlignedWithOther(
    const FlatAffineValueConstraints &other) {
  return areVarsAligned(*this, other);
}

/// Checks if the SSA values associated with `cst`'s variables in range
/// [start, end) are unique.
static bool LLVM_ATTRIBUTE_UNUSED areVarsUnique(
    const FlatAffineValueConstraints &cst, unsigned start, unsigned end) {

  assert(start <= cst.getNumDimAndSymbolVars() &&
         "Start position out of bounds");
  assert(end <= cst.getNumDimAndSymbolVars() && "End position out of bounds");

  if (start >= end)
    return true;

  SmallPtrSet<Value, 8> uniqueVars;
  ArrayRef<Optional<Value>> maybeValues =
      cst.getMaybeValues().slice(start, end - start);
  for (Optional<Value> val : maybeValues) {
    if (val && !uniqueVars.insert(*val).second)
      return false;
  }
  return true;
}

/// Checks if the SSA values associated with `cst`'s variables are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areVarsUnique(const FlatAffineValueConstraints &cst) {
  return areVarsUnique(cst, 0, cst.getNumDimAndSymbolVars());
}

/// Checks if the SSA values associated with `cst`'s variables of kind `kind`
/// are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areVarsUnique(const FlatAffineValueConstraints &cst, VarKind kind) {

  if (kind == VarKind::SetDim)
    return areVarsUnique(cst, 0, cst.getNumDimVars());
  if (kind == VarKind::Symbol)
    return areVarsUnique(cst, cst.getNumDimVars(),
                         cst.getNumDimAndSymbolVars());
  llvm_unreachable("Unexpected VarKind");
}

/// Merge and align the variables of A and B starting at 'offset', so that
/// both constraint systems get the union of the contained variables that is
/// dimension-wise and symbol-wise unique; both constraint systems are updated
/// so that they have the union of all variables, with A's original
/// variables appearing first followed by any of B's variables that didn't
/// appear in A. Local variables in B that have the same division
/// representation as local variables in A are merged into one.
//  E.g.: Input: A has ((%i, %j) [%M, %N]) and B has (%k, %j) [%P, %N, %M])
//        Output: both A, B have (%i, %j, %k) [%M, %N, %P]
static void mergeAndAlignVars(unsigned offset, FlatAffineValueConstraints *a,
                              FlatAffineValueConstraints *b) {
  assert(offset <= a->getNumDimVars() && offset <= b->getNumDimVars());
  // A merge/align isn't meaningful if a cst's vars aren't distinct.
  assert(areVarsUnique(*a) && "A's values aren't unique");
  assert(areVarsUnique(*b) && "B's values aren't unique");

  assert(
      llvm::all_of(llvm::drop_begin(a->getMaybeValues(), offset),
                   [](const Optional<Value> &var) { return var.has_value(); }));

  assert(
      llvm::all_of(llvm::drop_begin(b->getMaybeValues(), offset),
                   [](const Optional<Value> &var) { return var.has_value(); }));

  SmallVector<Value, 4> aDimValues;
  a->getValues(offset, a->getNumDimVars(), &aDimValues);

  {
    // Merge dims from A into B.
    unsigned d = offset;
    for (auto aDimValue : aDimValues) {
      unsigned loc;
      if (b->findVar(aDimValue, &loc)) {
        assert(loc >= offset && "A's dim appears in B's aligned range");
        assert(loc < b->getNumDimVars() &&
               "A's dim appears in B's non-dim position");
        b->swapVar(d, loc);
      } else {
        b->insertDimVar(d, aDimValue);
      }
      d++;
    }
    // Dimensions that are in B, but not in A, are added at the end.
    for (unsigned t = a->getNumDimVars(), e = b->getNumDimVars(); t < e; t++) {
      a->appendDimVar(b->getValue(t));
    }
    assert(a->getNumDimVars() == b->getNumDimVars() &&
           "expected same number of dims");
  }

  // Merge and align symbols of A and B
  a->mergeSymbolVars(*b);
  // Merge and align locals of A and B
  a->mergeLocalVars(*b);

  assert(areVarsAligned(*a, *b) && "IDs expected to be aligned");
}

// Call 'mergeAndAlignVars' to align constraint systems of 'this' and 'other'.
void FlatAffineValueConstraints::mergeAndAlignVarsWithOther(
    unsigned offset, FlatAffineValueConstraints *other) {
  mergeAndAlignVars(offset, this, other);
}

LogicalResult
FlatAffineValueConstraints::composeMap(const AffineValueMap *vMap) {
  return composeMatchingMap(
      computeAlignedMap(vMap->getAffineMap(), vMap->getOperands()));
}

// Similar to `composeMap` except that no Values need be associated with the
// constraint system nor are they looked at -- the dimensions and symbols of
// `other` are expected to correspond 1:1 to `this` system.
LogicalResult FlatAffineValueConstraints::composeMatchingMap(AffineMap other) {
  assert(other.getNumDims() == getNumDimVars() && "dim mismatch");
  assert(other.getNumSymbols() == getNumSymbolVars() && "symbol mismatch");

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(flattenAlignedMapAndMergeLocals(other, &flatExprs)))
    return failure();
  assert(flatExprs.size() == other.getNumResults());

  // Add dimensions corresponding to the map's results.
  insertDimVar(/*pos=*/0, /*num=*/other.getNumResults());

  // We add one equality for each result connecting the result dim of the map to
  // the other variables.
  // E.g.: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  // d_r - 16*i0 - i1 = 0. Similarly, when flattening (i0 + 1, i0 + 8*i2), we
  // add two equalities: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= other.getNumInputs() + 1);

    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, f = other.getNumInputs(); i < f; i++) {
      // Negate `eq[r]` since the newly added dimension will be set to this one.
      eqToAdd[e + i] = -flatExpr[i];
    }
    // Local columns of `eq` are at the beginning.
    unsigned j = getNumDimVars() + getNumSymbolVars();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = other.getNumInputs(); i < end; i++, j++) {
      eqToAdd[j] = -flatExpr[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -flatExpr[flatExpr.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }

  return success();
}

// Turn a symbol into a dimension.
static void turnSymbolIntoDim(FlatAffineValueConstraints *cst, Value value) {
  unsigned pos;
  if (cst->findVar(value, &pos) && pos >= cst->getNumDimVars() &&
      pos < cst->getNumDimAndSymbolVars()) {
    cst->swapVar(pos, cst->getNumDimVars());
    cst->setDimSymbolSeparation(cst->getNumSymbolVars() - 1);
  }
}

/// Merge and align symbols of `this` and `other` such that both get union of
/// of symbols that are unique. Symbols in `this` and `other` should be
/// unique. Symbols with Value as `None` are considered to be inequal to all
/// other symbols.
void FlatAffineValueConstraints::mergeSymbolVars(
    FlatAffineValueConstraints &other) {

  assert(areVarsUnique(*this, VarKind::Symbol) && "Symbol vars are not unique");
  assert(areVarsUnique(other, VarKind::Symbol) && "Symbol vars are not unique");

  SmallVector<Value, 4> aSymValues;
  getValues(getNumDimVars(), getNumDimAndSymbolVars(), &aSymValues);

  // Merge symbols: merge symbols into `other` first from `this`.
  unsigned s = other.getNumDimVars();
  for (Value aSymValue : aSymValues) {
    unsigned loc;
    // If the var is a symbol in `other`, then align it, otherwise assume that
    // it is a new symbol
    if (other.findVar(aSymValue, &loc) && loc >= other.getNumDimVars() &&
        loc < other.getNumDimAndSymbolVars())
      other.swapVar(s, loc);
    else
      other.insertSymbolVar(s - other.getNumDimVars(), aSymValue);
    s++;
  }

  // Symbols that are in other, but not in this, are added at the end.
  for (unsigned t = other.getNumDimVars() + getNumSymbolVars(),
                e = other.getNumDimAndSymbolVars();
       t < e; t++)
    insertSymbolVar(getNumSymbolVars(), other.getValue(t));

  assert(getNumSymbolVars() == other.getNumSymbolVars() &&
         "expected same number of symbols");
  assert(areVarsUnique(*this, VarKind::Symbol) && "Symbol vars are not unique");
  assert(areVarsUnique(other, VarKind::Symbol) && "Symbol vars are not unique");
}

// Changes all symbol variables which are loop IVs to dim variables.
void FlatAffineValueConstraints::convertLoopIVSymbolsToDims() {
  // Gather all symbols which are loop IVs.
  SmallVector<Value, 4> loopIVs;
  for (unsigned i = getNumDimVars(), e = getNumDimAndSymbolVars(); i < e; i++) {
    if (hasValue(i) && getForInductionVarOwner(getValue(i)))
      loopIVs.push_back(getValue(i));
  }
  // Turn each symbol in 'loopIVs' into a dim variable.
  for (auto iv : loopIVs) {
    turnSymbolIntoDim(this, iv);
  }
}

void FlatAffineValueConstraints::addInductionVarOrTerminalSymbol(Value val) {
  if (containsVar(val))
    return;

  // Caller is expected to fully compose map/operands if necessary.
  assert((isTopLevelValue(val) || isAffineForInductionVar(val)) &&
         "non-terminal symbol / loop IV expected");
  // Outer loop IVs could be used in forOp's bounds.
  if (auto loop = getForInductionVarOwner(val)) {
    appendDimVar(val);
    if (failed(this->addAffineForOpDomain(loop)))
      LLVM_DEBUG(
          loop.emitWarning("failed to add domain info to constraint system"));
    return;
  }
  // Add top level symbol.
  appendSymbolVar(val);
  // Check if the symbol is a constant.
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    addBound(BoundType::EQ, val, constOp.value());
}

LogicalResult
FlatAffineValueConstraints::addAffineForOpDomain(AffineForOp forOp) {
  unsigned pos;
  // Pre-condition for this method.
  if (!findVar(forOp.getInductionVar(), &pos)) {
    assert(false && "Value not found");
    return failure();
  }

  int64_t step = forOp.getStep();
  if (step != 1) {
    if (!forOp.hasConstantLowerBound())
      LLVM_DEBUG(forOp.emitWarning("domain conservatively approximated"));
    else {
      // Add constraints for the stride.
      // (iv - lb) % step = 0 can be written as:
      // (iv - lb) - step * q = 0 where q = (iv - lb) / step.
      // Add local variable 'q' and add the above equality.
      // The first constraint is q = (iv - lb) floordiv step
      SmallVector<int64_t, 8> dividend(getNumCols(), 0);
      int64_t lb = forOp.getConstantLowerBound();
      dividend[pos] = 1;
      dividend.back() -= lb;
      addLocalFloorDiv(dividend, step);
      // Second constraint: (iv - lb) - step * q = 0.
      SmallVector<int64_t, 8> eq(getNumCols(), 0);
      eq[pos] = 1;
      eq.back() -= lb;
      // For the local var just added above.
      eq[getNumCols() - 2] = -step;
      addEquality(eq);
    }
  }

  if (forOp.hasConstantLowerBound()) {
    addBound(BoundType::LB, pos, forOp.getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    if (failed(addBound(BoundType::LB, pos, forOp.getLowerBoundMap(),
                        forOp.getLowerBoundOperands())))
      return failure();
  }

  if (forOp.hasConstantUpperBound()) {
    addBound(BoundType::UB, pos, forOp.getConstantUpperBound() - 1);
    return success();
  }
  // Non-constant upper bound case.
  return addBound(BoundType::UB, pos, forOp.getUpperBoundMap(),
                  forOp.getUpperBoundOperands());
}

LogicalResult FlatAffineValueConstraints::addAffineParallelOpDomain(
    AffineParallelOp parallelOp) {
  size_t ivPos = 0;
  for (auto iv : parallelOp.getIVs()) {
    unsigned pos;
    if (!findVar(iv, &pos)) {
      assert(false && "variable expected for the IV value");
      return failure();
    }

    AffineMap lowerBound = parallelOp.getLowerBoundMap(ivPos);
    if (lowerBound.isConstant())
      addBound(BoundType::LB, pos, lowerBound.getSingleConstantResult());
    else if (failed(addBound(BoundType::LB, pos, lowerBound,
                             parallelOp.getLowerBoundsOperands())))
      return failure();

    auto upperBound = parallelOp.getUpperBoundMap(ivPos);
    if (upperBound.isConstant())
      addBound(BoundType::UB, pos, upperBound.getSingleConstantResult());
    else if (failed(addBound(BoundType::UB, pos, upperBound,
                             parallelOp.getUpperBoundsOperands())))
      return failure();
  }
  return success();
}

LogicalResult
FlatAffineValueConstraints::addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                                   ArrayRef<AffineMap> ubMaps,
                                                   ArrayRef<Value> operands) {
  assert(lbMaps.size() == ubMaps.size());
  assert(lbMaps.size() <= getNumDimVars());

  for (unsigned i = 0, e = lbMaps.size(); i < e; ++i) {
    AffineMap lbMap = lbMaps[i];
    AffineMap ubMap = ubMaps[i];
    assert(!lbMap || lbMap.getNumInputs() == operands.size());
    assert(!ubMap || ubMap.getNumInputs() == operands.size());

    // Check if this slice is just an equality along this dimension. If so,
    // retrieve the existing loop it equates to and add it to the system.
    if (lbMap && ubMap && lbMap.getNumResults() == 1 &&
        ubMap.getNumResults() == 1 &&
        lbMap.getResult(0) + 1 == ubMap.getResult(0) &&
        // The condition above will be true for maps describing a single
        // iteration (e.g., lbMap.getResult(0) = 0, ubMap.getResult(0) = 1).
        // Make sure we skip those cases by checking that the lb result is not
        // just a constant.
        !lbMap.getResult(0).isa<AffineConstantExpr>()) {
      // Limited support: we expect the lb result to be just a loop dimension.
      // Not supported otherwise for now.
      AffineDimExpr result = lbMap.getResult(0).dyn_cast<AffineDimExpr>();
      if (!result)
        return failure();

      AffineForOp loop =
          getForInductionVarOwner(operands[result.getPosition()]);
      if (!loop)
        return failure();

      if (failed(addAffineForOpDomain(loop)))
        return failure();
      continue;
    }

    // This slice refers to a loop that doesn't exist in the IR yet. Add its
    // bounds to the system assuming its dimension variable position is the
    // same as the position of the loop in the loop nest.
    if (lbMap && failed(addBound(BoundType::LB, i, lbMap, operands)))
      return failure();
    if (ubMap && failed(addBound(BoundType::UB, i, ubMap, operands)))
      return failure();
  }
  return success();
}

void FlatAffineValueConstraints::addAffineIfOpDomain(AffineIfOp ifOp) {
  IntegerSet set = ifOp.getIntegerSet();
  // Canonicalize set and operands to ensure unique values for
  // FlatAffineValueConstraints below and for early simplification.
  SmallVector<Value> operands(ifOp.getOperands());
  canonicalizeSetAndOperands(&set, &operands);

  // Create the base constraints from the integer set attached to ifOp.
  FlatAffineValueConstraints cst(set, operands);

  // Merge the constraints from ifOp to the current domain. We need first merge
  // and align the IDs from both constraints, and then append the constraints
  // from the ifOp into the current one.
  mergeAndAlignVarsWithOther(0, &cst);
  append(cst);
}

bool FlatAffineValueConstraints::hasConsistentState() const {
  return IntegerPolyhedron::hasConsistentState() &&
         values.size() == getNumDimAndSymbolVars();
}

void FlatAffineValueConstraints::removeVarRange(VarKind kind, unsigned varStart,
                                                unsigned varLimit) {
  IntegerPolyhedron::removeVarRange(kind, varStart, varLimit);
  unsigned offset = getVarKindOffset(kind);

  if (kind != VarKind::Local) {
    values.erase(values.begin() + varStart + offset,
                 values.begin() + varLimit + offset);
  }
}

// Determine whether the variable at 'pos' (say var_r) can be expressed as
// modulo of another known variable (say var_n) w.r.t a constant. For example,
// if the following constraints hold true:
// ```
// 0 <= var_r <= divisor - 1
// var_n - (divisor * q_expr) = var_r
// ```
// where `var_n` is a known variable (called dividend), and `q_expr` is an
// `AffineExpr` (called the quotient expression), `var_r` can be written as:
//
// `var_r = var_n mod divisor`.
//
// Additionally, in a special case of the above constaints where `q_expr` is an
// variable itself that is not yet known (say `var_q`), it can be written as a
// floordiv in the following way:
//
// `var_q = var_n floordiv divisor`.
//
// Returns true if the above mod or floordiv are detected, updating 'memo' with
// these new expressions. Returns false otherwise.
static bool detectAsMod(const FlatAffineValueConstraints &cst, unsigned pos,
                        int64_t lbConst, int64_t ubConst,
                        SmallVectorImpl<AffineExpr> &memo,
                        MLIRContext *context) {
  assert(pos < cst.getNumVars() && "invalid position");

  // Check if a divisor satisfying the condition `0 <= var_r <= divisor - 1` can
  // be determined.
  if (lbConst != 0 || ubConst < 1)
    return false;
  int64_t divisor = ubConst + 1;

  // Check for the aforementioned conditions in each equality.
  for (unsigned curEquality = 0, numEqualities = cst.getNumEqualities();
       curEquality < numEqualities; curEquality++) {
    int64_t coefficientAtPos = cst.atEq64(curEquality, pos);
    // If current equality does not involve `var_r`, continue to the next
    // equality.
    if (coefficientAtPos == 0)
      continue;

    // Constant term should be 0 in this equality.
    if (cst.atEq64(curEquality, cst.getNumCols() - 1) != 0)
      continue;

    // Traverse through the equality and construct the dividend expression
    // `dividendExpr`, to contain all the variables which are known and are
    // not divisible by `(coefficientAtPos * divisor)`. Hope here is that the
    // `dividendExpr` gets simplified into a single variable `var_n` discussed
    // above.
    auto dividendExpr = getAffineConstantExpr(0, context);

    // Track the terms that go into quotient expression, later used to detect
    // additional floordiv.
    unsigned quotientCount = 0;
    int quotientPosition = -1;
    int quotientSign = 1;

    // Consider each term in the current equality.
    unsigned curVar, e;
    for (curVar = 0, e = cst.getNumDimAndSymbolVars(); curVar < e; ++curVar) {
      // Ignore var_r.
      if (curVar == pos)
        continue;
      int64_t coefficientOfCurVar = cst.atEq64(curEquality, curVar);
      // Ignore vars that do not contribute to the current equality.
      if (coefficientOfCurVar == 0)
        continue;
      // Check if the current var goes into the quotient expression.
      if (coefficientOfCurVar % (divisor * coefficientAtPos) == 0) {
        quotientCount++;
        quotientPosition = curVar;
        quotientSign = (coefficientOfCurVar * coefficientAtPos) > 0 ? 1 : -1;
        continue;
      }
      // Variables that are part of dividendExpr should be known.
      if (!memo[curVar])
        break;
      // Append the current variable to the dividend expression.
      dividendExpr = dividendExpr + memo[curVar] * coefficientOfCurVar;
    }

    // Can't construct expression as it depends on a yet uncomputed var.
    if (curVar < e)
      continue;

    // Express `var_r` in terms of the other vars collected so far.
    if (coefficientAtPos > 0)
      dividendExpr = (-dividendExpr).floorDiv(coefficientAtPos);
    else
      dividendExpr = dividendExpr.floorDiv(-coefficientAtPos);

    // Simplify the expression.
    dividendExpr = simplifyAffineExpr(dividendExpr, cst.getNumDimVars(),
                                      cst.getNumSymbolVars());
    // Only if the final dividend expression is just a single var (which we call
    // `var_n`), we can proceed.
    // TODO: Handle AffineSymbolExpr as well. There is no reason to restrict it
    // to dims themselves.
    auto dimExpr = dividendExpr.dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      continue;

    // Express `var_r` as `var_n % divisor` and store the expression in `memo`.
    if (quotientCount >= 1) {
      auto ub = cst.getConstantBound64(
          FlatAffineValueConstraints::BoundType::UB, dimExpr.getPosition());
      // If `var_n` has an upperbound that is less than the divisor, mod can be
      // eliminated altogether.
      if (ub && *ub < divisor)
        memo[pos] = dimExpr;
      else
        memo[pos] = dimExpr % divisor;
      // If a unique quotient `var_q` was seen, it can be expressed as
      // `var_n floordiv divisor`.
      if (quotientCount == 1 && !memo[quotientPosition])
        memo[quotientPosition] = dimExpr.floorDiv(divisor) * quotientSign;

      return true;
    }
  }
  return false;
}

/// Check if the pos^th variable can be expressed as a floordiv of an affine
/// function of other variables (where the divisor is a positive constant)
/// given the initial set of expressions in `exprs`. If it can be, the
/// corresponding position in `exprs` is set as the detected affine expr. For
/// eg: 4q <= i + j <= 4q + 3   <=>   q = (i + j) floordiv 4. An equality can
/// also yield a floordiv: eg.  4q = i + j <=> q = (i + j) floordiv 4. 32q + 28
/// <= i <= 32q + 31 => q = i floordiv 32.
static bool detectAsFloorDiv(const FlatAffineValueConstraints &cst,
                             unsigned pos, MLIRContext *context,
                             SmallVectorImpl<AffineExpr> &exprs) {
  assert(pos < cst.getNumVars() && "invalid position");

  // Get upper-lower bound pair for this variable.
  SmallVector<bool, 8> foundRepr(cst.getNumVars(), false);
  for (unsigned i = 0, e = cst.getNumVars(); i < e; ++i)
    if (exprs[i])
      foundRepr[i] = true;

  SmallVector<int64_t, 8> dividend(cst.getNumCols());
  unsigned divisor;
  auto ulPair = computeSingleVarRepr(cst, foundRepr, pos, dividend, divisor);

  // No upper-lower bound pair found for this var.
  if (ulPair.kind == ReprKind::None || ulPair.kind == ReprKind::Equality)
    return false;

  // Construct the dividend expression.
  auto dividendExpr = getAffineConstantExpr(dividend.back(), context);
  for (unsigned c = 0, f = cst.getNumVars(); c < f; c++)
    if (dividend[c] != 0)
      dividendExpr = dividendExpr + dividend[c] * exprs[c];

  // Successfully detected the floordiv.
  exprs[pos] = dividendExpr.floorDiv(divisor);
  return true;
}

std::pair<AffineMap, AffineMap>
FlatAffineValueConstraints::getLowerAndUpperBound(
    unsigned pos, unsigned offset, unsigned num, unsigned symStartPos,
    ArrayRef<AffineExpr> localExprs, MLIRContext *context) const {
  assert(pos + offset < getNumDimVars() && "invalid dim start pos");
  assert(symStartPos >= (pos + offset) && "invalid sym start pos");
  assert(getNumLocalVars() == localExprs.size() &&
         "incorrect local exprs count");

  SmallVector<unsigned, 4> lbIndices, ubIndices, eqIndices;
  getLowerAndUpperBoundIndices(pos + offset, &lbIndices, &ubIndices, &eqIndices,
                               offset, num);

  /// Add to 'b' from 'a' in set [0, offset) U [offset + num, symbStartPos).
  auto addCoeffs = [&](ArrayRef<int64_t> a, SmallVectorImpl<int64_t> &b) {
    b.clear();
    for (unsigned i = 0, e = a.size(); i < e; ++i) {
      if (i < offset || i >= offset + num)
        b.push_back(a[i]);
    }
  };

  SmallVector<int64_t, 8> lb, ub;
  SmallVector<AffineExpr, 4> lbExprs;
  unsigned dimCount = symStartPos - num;
  unsigned symCount = getNumDimAndSymbolVars() - symStartPos;
  lbExprs.reserve(lbIndices.size() + eqIndices.size());
  // Lower bound expressions.
  for (auto idx : lbIndices) {
    auto ineq = getInequality64(idx);
    // Extract the lower bound (in terms of other coeff's + const), i.e., if
    // i - j + 1 >= 0 is the constraint, 'pos' is for i the lower bound is j
    // - 1.
    addCoeffs(ineq, lb);
    std::transform(lb.begin(), lb.end(), lb.begin(), std::negate<int64_t>());
    auto expr =
        getAffineExprFromFlatForm(lb, dimCount, symCount, localExprs, context);
    // expr ceildiv divisor is (expr + divisor - 1) floordiv divisor
    int64_t divisor = std::abs(ineq[pos + offset]);
    expr = (expr + divisor - 1).floorDiv(divisor);
    lbExprs.push_back(expr);
  }

  SmallVector<AffineExpr, 4> ubExprs;
  ubExprs.reserve(ubIndices.size() + eqIndices.size());
  // Upper bound expressions.
  for (auto idx : ubIndices) {
    auto ineq = getInequality64(idx);
    // Extract the upper bound (in terms of other coeff's + const).
    addCoeffs(ineq, ub);
    auto expr =
        getAffineExprFromFlatForm(ub, dimCount, symCount, localExprs, context);
    expr = expr.floorDiv(std::abs(ineq[pos + offset]));
    // Upper bound is exclusive.
    ubExprs.push_back(expr + 1);
  }

  // Equalities. It's both a lower and a upper bound.
  SmallVector<int64_t, 4> b;
  for (auto idx : eqIndices) {
    auto eq = getEquality64(idx);
    addCoeffs(eq, b);
    if (eq[pos + offset] > 0)
      std::transform(b.begin(), b.end(), b.begin(), std::negate<int64_t>());

    // Extract the upper bound (in terms of other coeff's + const).
    auto expr =
        getAffineExprFromFlatForm(b, dimCount, symCount, localExprs, context);
    expr = expr.floorDiv(std::abs(eq[pos + offset]));
    // Upper bound is exclusive.
    ubExprs.push_back(expr + 1);
    // Lower bound.
    expr =
        getAffineExprFromFlatForm(b, dimCount, symCount, localExprs, context);
    expr = expr.ceilDiv(std::abs(eq[pos + offset]));
    lbExprs.push_back(expr);
  }

  auto lbMap = AffineMap::get(dimCount, symCount, lbExprs, context);
  auto ubMap = AffineMap::get(dimCount, symCount, ubExprs, context);

  return {lbMap, ubMap};
}

/// Computes the lower and upper bounds of the first 'num' dimensional
/// variables (starting at 'offset') as affine maps of the remaining
/// variables (dimensional and symbolic variables). Local variables are
/// themselves explicitly computed as affine functions of other variables in
/// this process if needed.
void FlatAffineValueConstraints::getSliceBounds(
    unsigned offset, unsigned num, MLIRContext *context,
    SmallVectorImpl<AffineMap> *lbMaps, SmallVectorImpl<AffineMap> *ubMaps,
    bool getClosedUB) {
  assert(num < getNumDimVars() && "invalid range");

  // Basic simplification.
  normalizeConstraintsByGCD();

  LLVM_DEBUG(llvm::dbgs() << "getSliceBounds for first " << num
                          << " variables\n");
  LLVM_DEBUG(dump());

  // Record computed/detected variables.
  SmallVector<AffineExpr, 8> memo(getNumVars());
  // Initialize dimensional and symbolic variables.
  for (unsigned i = 0, e = getNumDimVars(); i < e; i++) {
    if (i < offset)
      memo[i] = getAffineDimExpr(i, context);
    else if (i >= offset + num)
      memo[i] = getAffineDimExpr(i - num, context);
  }
  for (unsigned i = getNumDimVars(), e = getNumDimAndSymbolVars(); i < e; i++)
    memo[i] = getAffineSymbolExpr(i - getNumDimVars(), context);

  bool changed;
  do {
    changed = false;
    // Identify yet unknown variables as constants or mod's / floordiv's of
    // other variables if possible.
    for (unsigned pos = 0; pos < getNumVars(); pos++) {
      if (memo[pos])
        continue;

      auto lbConst = getConstantBound64(BoundType::LB, pos);
      auto ubConst = getConstantBound64(BoundType::UB, pos);
      if (lbConst.has_value() && ubConst.has_value()) {
        // Detect equality to a constant.
        if (*lbConst == *ubConst) {
          memo[pos] = getAffineConstantExpr(*lbConst, context);
          changed = true;
          continue;
        }

        // Detect an variable as modulo of another variable w.r.t a
        // constant.
        if (detectAsMod(*this, pos, *lbConst, *ubConst, memo, context)) {
          changed = true;
          continue;
        }
      }

      // Detect an variable as a floordiv of an affine function of other
      // variables (divisor is a positive constant).
      if (detectAsFloorDiv(*this, pos, context, memo)) {
        changed = true;
        continue;
      }

      // Detect an variable as an expression of other variables.
      unsigned idx;
      if (!findConstraintWithNonZeroAt(pos, /*isEq=*/true, &idx)) {
        continue;
      }

      // Build AffineExpr solving for variable 'pos' in terms of all others.
      auto expr = getAffineConstantExpr(0, context);
      unsigned j, e;
      for (j = 0, e = getNumVars(); j < e; ++j) {
        if (j == pos)
          continue;
        int64_t c = atEq64(idx, j);
        if (c == 0)
          continue;
        // If any of the involved IDs hasn't been found yet, we can't proceed.
        if (!memo[j])
          break;
        expr = expr + memo[j] * c;
      }
      if (j < e)
        // Can't construct expression as it depends on a yet uncomputed
        // variable.
        continue;

      // Add constant term to AffineExpr.
      expr = expr + atEq64(idx, getNumVars());
      int64_t vPos = atEq64(idx, pos);
      assert(vPos != 0 && "expected non-zero here");
      if (vPos > 0)
        expr = (-expr).floorDiv(vPos);
      else
        // vPos < 0.
        expr = expr.floorDiv(-vPos);
      // Successfully constructed expression.
      memo[pos] = expr;
      changed = true;
    }
    // This loop is guaranteed to reach a fixed point - since once an
    // variable's explicit form is computed (in memo[pos]), it's not updated
    // again.
  } while (changed);

  int64_t ubAdjustment = getClosedUB ? 0 : 1;

  // Set the lower and upper bound maps for all the variables that were
  // computed as affine expressions of the rest as the "detected expr" and
  // "detected expr + 1" respectively; set the undetected ones to null.
  std::optional<FlatAffineValueConstraints> tmpClone;
  for (unsigned pos = 0; pos < num; pos++) {
    unsigned numMapDims = getNumDimVars() - num;
    unsigned numMapSymbols = getNumSymbolVars();
    AffineExpr expr = memo[pos + offset];
    if (expr)
      expr = simplifyAffineExpr(expr, numMapDims, numMapSymbols);

    AffineMap &lbMap = (*lbMaps)[pos];
    AffineMap &ubMap = (*ubMaps)[pos];

    if (expr) {
      lbMap = AffineMap::get(numMapDims, numMapSymbols, expr);
      ubMap = AffineMap::get(numMapDims, numMapSymbols, expr + ubAdjustment);
    } else {
      // TODO: Whenever there are local variables in the dependence
      // constraints, we'll conservatively over-approximate, since we don't
      // always explicitly compute them above (in the while loop).
      if (getNumLocalVars() == 0) {
        // Work on a copy so that we don't update this constraint system.
        if (!tmpClone) {
          tmpClone.emplace(FlatAffineValueConstraints(*this));
          // Removing redundant inequalities is necessary so that we don't get
          // redundant loop bounds.
          tmpClone->removeRedundantInequalities();
        }
        std::tie(lbMap, ubMap) = tmpClone->getLowerAndUpperBound(
            pos, offset, num, getNumDimVars(), /*localExprs=*/{}, context);
      }

      // If the above fails, we'll just use the constant lower bound and the
      // constant upper bound (if they exist) as the slice bounds.
      // TODO: being conservative for the moment in cases that
      // lead to multiple bounds - until getConstDifference in LoopFusion.cpp is
      // fixed (b/126426796).
      if (!lbMap || lbMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice lb\n");
        auto lbConst = getConstantBound64(BoundType::LB, pos + offset);
        if (lbConst.has_value()) {
          lbMap = AffineMap::get(numMapDims, numMapSymbols,
                                 getAffineConstantExpr(*lbConst, context));
        }
      }
      if (!ubMap || ubMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice ub\n");
        auto ubConst = getConstantBound64(BoundType::UB, pos + offset);
        if (ubConst.has_value()) {
          ubMap = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(*ubConst + ubAdjustment, context));
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "lb map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(lbMap.dump(););
    LLVM_DEBUG(llvm::dbgs()
               << "ub map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(ubMap.dump(););
  }
}

LogicalResult FlatAffineValueConstraints::flattenAlignedMapAndMergeLocals(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs) {
  FlatAffineValueConstraints localCst;
  if (failed(getFlattenedAffineExprs(map, flattenedExprs, &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }

  // Add localCst information.
  if (localCst.getNumLocalVars() > 0) {
    unsigned numLocalVars = getNumLocalVars();
    // Insert local dims of localCst at the beginning.
    insertLocalVar(/*pos=*/0, /*num=*/localCst.getNumLocalVars());
    // Insert local dims of `this` at the end of localCst.
    localCst.appendLocalVar(/*num=*/numLocalVars);
    // Dimensions of localCst and this constraint set match. Append localCst to
    // this constraint set.
    append(localCst);
  }

  return success();
}

LogicalResult FlatAffineValueConstraints::addBound(BoundType type, unsigned pos,
                                                   AffineMap boundMap,
                                                   bool isClosedBound) {
  assert(boundMap.getNumDims() == getNumDimVars() && "dim mismatch");
  assert(boundMap.getNumSymbols() == getNumSymbolVars() && "symbol mismatch");
  assert(pos < getNumDimAndSymbolVars() && "invalid position");
  assert((type != BoundType::EQ || isClosedBound) &&
         "EQ bound must be closed.");

  // Equality follows the logic of lower bound except that we add an equality
  // instead of an inequality.
  assert((type != BoundType::EQ || boundMap.getNumResults() == 1) &&
         "single result expected");
  bool lower = type == BoundType::LB || type == BoundType::EQ;

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(flattenAlignedMapAndMergeLocals(boundMap, &flatExprs)))
    return failure();
  assert(flatExprs.size() == boundMap.getNumResults());

  // Add one (in)equality for each result.
  for (const auto &flatExpr : flatExprs) {
    SmallVector<int64_t> ineq(getNumCols(), 0);
    // Dims and symbols.
    for (unsigned j = 0, e = boundMap.getNumInputs(); j < e; j++) {
      ineq[j] = lower ? -flatExpr[j] : flatExpr[j];
    }
    // Invalid bound: pos appears in `boundMap`.
    // TODO: This should be an assertion. Fix `addDomainFromSliceMaps` and/or
    // its callers to prevent invalid bounds from being added.
    if (ineq[pos] != 0)
      continue;
    ineq[pos] = lower ? 1 : -1;
    // Local columns of `ineq` are at the beginning.
    unsigned j = getNumDimVars() + getNumSymbolVars();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = boundMap.getNumInputs(); i < end; i++, j++) {
      ineq[j] = lower ? -flatExpr[i] : flatExpr[i];
    }
    // Make the bound closed in if flatExpr is open. The inequality is always
    // created in the upper bound form, so the adjustment is -1.
    int64_t boundAdjustment = (isClosedBound || type == BoundType::EQ) ? 0 : -1;
    // Constant term.
    ineq[getNumCols() - 1] = (lower ? -flatExpr[flatExpr.size() - 1]
                                    : flatExpr[flatExpr.size() - 1]) +
                             boundAdjustment;
    type == BoundType::EQ ? addEquality(ineq) : addInequality(ineq);
  }

  return success();
}

LogicalResult FlatAffineValueConstraints::addBound(BoundType type, unsigned pos,
                                                   AffineMap boundMap) {
  return addBound(type, pos, boundMap, /*isClosedBound=*/type != BoundType::UB);
}

AffineMap
FlatAffineValueConstraints::computeAlignedMap(AffineMap map,
                                              ValueRange operands) const {
  assert(map.getNumInputs() == operands.size() && "number of inputs mismatch");

  SmallVector<Value> dims, syms;
#ifndef NDEBUG
  SmallVector<Value> newSyms;
  SmallVector<Value> *newSymsPtr = &newSyms;
#else
  SmallVector<Value> *newSymsPtr = nullptr;
#endif // NDEBUG

  dims.reserve(getNumDimVars());
  syms.reserve(getNumSymbolVars());
  for (unsigned i = getVarKindOffset(VarKind::SetDim),
                e = getVarKindEnd(VarKind::SetDim);
       i < e; ++i)
    dims.push_back(values[i] ? *values[i] : Value());
  for (unsigned i = getVarKindOffset(VarKind::Symbol),
                e = getVarKindEnd(VarKind::Symbol);
       i < e; ++i)
    syms.push_back(values[i] ? *values[i] : Value());

  AffineMap alignedMap =
      alignAffineMapWithValues(map, operands, dims, syms, newSymsPtr);
  // All symbols are already part of this FlatAffineConstraints.
  assert(syms.size() == newSymsPtr->size() && "unexpected new/missing symbols");
  assert(std::equal(syms.begin(), syms.end(), newSymsPtr->begin()) &&
         "unexpected new/missing symbols");
  return alignedMap;
}

LogicalResult FlatAffineValueConstraints::addBound(BoundType type, unsigned pos,
                                                   AffineMap boundMap,
                                                   ValueRange boundOperands) {
  // Fully compose map and operands; canonicalize and simplify so that we
  // transitively get to terminal symbols or loop IVs.
  auto map = boundMap;
  SmallVector<Value, 4> operands(boundOperands.begin(), boundOperands.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  for (auto operand : operands)
    addInductionVarOrTerminalSymbol(operand);
  return addBound(type, pos, computeAlignedMap(map, operands));
}

// Adds slice lower bounds represented by lower bounds in 'lbMaps' and upper
// bounds in 'ubMaps' to each value in `values' that appears in the constraint
// system. Note that both lower/upper bounds share the same operand list
// 'operands'.
// This function assumes 'values.size' == 'lbMaps.size' == 'ubMaps.size', and
// skips any null AffineMaps in 'lbMaps' or 'ubMaps'.
// Note that both lower/upper bounds use operands from 'operands'.
// Returns failure for unimplemented cases such as semi-affine expressions or
// expressions with mod/floordiv.
LogicalResult FlatAffineValueConstraints::addSliceBounds(
    ArrayRef<Value> values, ArrayRef<AffineMap> lbMaps,
    ArrayRef<AffineMap> ubMaps, ArrayRef<Value> operands) {
  assert(values.size() == lbMaps.size());
  assert(lbMaps.size() == ubMaps.size());

  for (unsigned i = 0, e = lbMaps.size(); i < e; ++i) {
    unsigned pos;
    if (!findVar(values[i], &pos))
      continue;

    AffineMap lbMap = lbMaps[i];
    AffineMap ubMap = ubMaps[i];
    assert(!lbMap || lbMap.getNumInputs() == operands.size());
    assert(!ubMap || ubMap.getNumInputs() == operands.size());

    // Check if this slice is just an equality along this dimension.
    if (lbMap && ubMap && lbMap.getNumResults() == 1 &&
        ubMap.getNumResults() == 1 &&
        lbMap.getResult(0) + 1 == ubMap.getResult(0)) {
      if (failed(addBound(BoundType::EQ, pos, lbMap, operands)))
        return failure();
      continue;
    }

    // If lower or upper bound maps are null or provide no results, it implies
    // that the source loop was not at all sliced, and the entire loop will be a
    // part of the slice.
    if (lbMap && lbMap.getNumResults() != 0 && ubMap &&
        ubMap.getNumResults() != 0) {
      if (failed(addBound(BoundType::LB, pos, lbMap, operands)))
        return failure();
      if (failed(addBound(BoundType::UB, pos, ubMap, operands)))
        return failure();
    } else {
      auto loop = getForInductionVarOwner(values[i]);
      if (failed(this->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

bool FlatAffineValueConstraints::findVar(Value val, unsigned *pos) const {
  unsigned i = 0;
  for (const auto &mayBeVar : values) {
    if (mayBeVar && *mayBeVar == val) {
      *pos = i;
      return true;
    }
    i++;
  }
  return false;
}

bool FlatAffineValueConstraints::containsVar(Value val) const {
  return llvm::any_of(values, [&](const Optional<Value> &mayBeVar) {
    return mayBeVar && *mayBeVar == val;
  });
}

void FlatAffineValueConstraints::swapVar(unsigned posA, unsigned posB) {
  IntegerPolyhedron::swapVar(posA, posB);

  if (getVarKindAt(posA) == VarKind::Local &&
      getVarKindAt(posB) == VarKind::Local)
    return;

  // Treat value of a local variable as None.
  if (getVarKindAt(posA) == VarKind::Local)
    values[posB] = std::nullopt;
  else if (getVarKindAt(posB) == VarKind::Local)
    values[posA] = std::nullopt;
  else
    std::swap(values[posA], values[posB]);
}

void FlatAffineValueConstraints::addBound(BoundType type, Value val,
                                          int64_t value) {
  unsigned pos;
  if (!findVar(val, &pos))
    // This is a pre-condition for this method.
    assert(0 && "var not found");
  addBound(type, pos, value);
}

void FlatAffineValueConstraints::printSpace(raw_ostream &os) const {
  IntegerPolyhedron::printSpace(os);
  os << "(";
  for (unsigned i = 0, e = getNumDimAndSymbolVars(); i < e; i++) {
    if (hasValue(i))
      os << "Value ";
    else
      os << "None ";
  }
  for (unsigned i = getVarKindOffset(VarKind::Local),
                e = getVarKindEnd(VarKind::Local);
       i < e; ++i)
    os << "Local ";
  os << " const)\n";
}

void FlatAffineValueConstraints::clearAndCopyFrom(
    const IntegerRelation &other) {

  if (auto *otherValueSet =
          dyn_cast<const FlatAffineValueConstraints>(&other)) {
    *this = *otherValueSet;
  } else {
    *static_cast<IntegerRelation *>(this) = other;
    values.clear();
    values.resize(getNumDimAndSymbolVars(), std::nullopt);
  }
}

void FlatAffineValueConstraints::fourierMotzkinEliminate(
    unsigned pos, bool darkShadow, bool *isResultIntegerExact) {
  SmallVector<Optional<Value>, 8> newVals = values;
  if (getVarKindAt(pos) != VarKind::Local)
    newVals.erase(newVals.begin() + pos);
  // Note: Base implementation discards all associated Values.
  IntegerPolyhedron::fourierMotzkinEliminate(pos, darkShadow,
                                             isResultIntegerExact);
  values = newVals;
  assert(values.size() == getNumDimAndSymbolVars());
}

void FlatAffineValueConstraints::projectOut(Value val) {
  unsigned pos;
  bool ret = findVar(val, &pos);
  assert(ret);
  (void)ret;
  fourierMotzkinEliminate(pos);
}

LogicalResult FlatAffineValueConstraints::unionBoundingBox(
    const FlatAffineValueConstraints &otherCst) {
  assert(otherCst.getNumDimVars() == getNumDimVars() && "dims mismatch");
  assert(otherCst.getMaybeValues()
             .slice(0, getNumDimVars())
             .equals(getMaybeValues().slice(0, getNumDimVars())) &&
         "dim values mismatch");
  assert(otherCst.getNumLocalVars() == 0 && "local vars not supported here");
  assert(getNumLocalVars() == 0 && "local vars not supported yet here");

  // Align `other` to this.
  if (!areVarsAligned(*this, otherCst)) {
    FlatAffineValueConstraints otherCopy(otherCst);
    mergeAndAlignVars(/*offset=*/getNumDimVars(), this, &otherCopy);
    return IntegerPolyhedron::unionBoundingBox(otherCopy);
  }

  return IntegerPolyhedron::unionBoundingBox(otherCst);
}

/// Compute an explicit representation for local vars. For all systems coming
/// from MLIR integer sets, maps, or expressions where local vars were
/// introduced to model floordivs and mods, this always succeeds.
static LogicalResult computeLocalVars(const FlatAffineValueConstraints &cst,
                                      SmallVectorImpl<AffineExpr> &memo,
                                      MLIRContext *context) {
  unsigned numDims = cst.getNumDimVars();
  unsigned numSyms = cst.getNumSymbolVars();

  // Initialize dimensional and symbolic variables.
  for (unsigned i = 0; i < numDims; i++)
    memo[i] = getAffineDimExpr(i, context);
  for (unsigned i = numDims, e = numDims + numSyms; i < e; i++)
    memo[i] = getAffineSymbolExpr(i - numDims, context);

  bool changed;
  do {
    // Each time `changed` is true at the end of this iteration, one or more
    // local vars would have been detected as floordivs and set in memo; so the
    // number of null entries in memo[...] strictly reduces; so this converges.
    changed = false;
    for (unsigned i = 0, e = cst.getNumLocalVars(); i < e; ++i)
      if (!memo[numDims + numSyms + i] &&
          detectAsFloorDiv(cst, /*pos=*/numDims + numSyms + i, context, memo))
        changed = true;
  } while (changed);

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(cst.getNumLocalVars());
  return success(
      llvm::all_of(localExprs, [](AffineExpr expr) { return expr; }));
}

void FlatAffineValueConstraints::getIneqAsAffineValueMap(
    unsigned pos, unsigned ineqPos, AffineValueMap &vmap,
    MLIRContext *context) const {
  unsigned numDims = getNumDimVars();
  unsigned numSyms = getNumSymbolVars();

  assert(pos < numDims && "invalid position");
  assert(ineqPos < getNumInequalities() && "invalid inequality position");

  // Get expressions for local vars.
  SmallVector<AffineExpr, 8> memo(getNumVars(), AffineExpr());
  if (failed(computeLocalVars(*this, memo, context)))
    assert(false &&
           "one or more local exprs do not have an explicit representation");
  auto localExprs = ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());

  // Compute the AffineExpr lower/upper bound for this inequality.
  SmallVector<int64_t, 8> inequality = getInequality64(ineqPos);
  SmallVector<int64_t, 8> bound;
  bound.reserve(getNumCols() - 1);
  // Everything other than the coefficient at `pos`.
  bound.append(inequality.begin(), inequality.begin() + pos);
  bound.append(inequality.begin() + pos + 1, inequality.end());

  if (inequality[pos] > 0)
    // Lower bound.
    std::transform(bound.begin(), bound.end(), bound.begin(),
                   std::negate<int64_t>());
  else
    // Upper bound (which is exclusive).
    bound.back() += 1;

  // Convert to AffineExpr (tree) form.
  auto boundExpr = getAffineExprFromFlatForm(bound, numDims - 1, numSyms,
                                             localExprs, context);

  // Get the values to bind to this affine expr (all dims and symbols).
  SmallVector<Value, 4> operands;
  getValues(0, pos, &operands);
  SmallVector<Value, 4> trailingOperands;
  getValues(pos + 1, getNumDimAndSymbolVars(), &trailingOperands);
  operands.append(trailingOperands.begin(), trailingOperands.end());
  vmap.reset(AffineMap::get(numDims - 1, numSyms, boundExpr), operands);
}

IntegerSet
FlatAffineValueConstraints::getAsIntegerSet(MLIRContext *context) const {
  if (getNumConstraints() == 0)
    // Return universal set (always true): 0 == 0.
    return IntegerSet::get(getNumDimVars(), getNumSymbolVars(),
                           getAffineConstantExpr(/*constant=*/0, context),
                           /*eqFlags=*/true);

  // Construct local references.
  SmallVector<AffineExpr, 8> memo(getNumVars(), AffineExpr());

  if (failed(computeLocalVars(*this, memo, context))) {
    // Check if the local variables without an explicit representation have
    // zero coefficients everywhere.
    SmallVector<unsigned> noLocalRepVars;
    unsigned numDimsSymbols = getNumDimAndSymbolVars();
    for (unsigned i = numDimsSymbols, e = getNumVars(); i < e; ++i) {
      if (!memo[i] && !isColZero(/*pos=*/i))
        noLocalRepVars.push_back(i - numDimsSymbols);
    }
    if (!noLocalRepVars.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "local variables at position(s) ";
        llvm::interleaveComma(noLocalRepVars, llvm::dbgs());
        llvm::dbgs() << " do not have an explicit representation in:\n";
        this->dump();
      });
      return IntegerSet();
    }
  }

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());

  // Construct the IntegerSet from the equalities/inequalities.
  unsigned numDims = getNumDimVars();
  unsigned numSyms = getNumSymbolVars();

  SmallVector<bool, 16> eqFlags(getNumConstraints());
  std::fill(eqFlags.begin(), eqFlags.begin() + getNumEqualities(), true);
  std::fill(eqFlags.begin() + getNumEqualities(), eqFlags.end(), false);

  SmallVector<AffineExpr, 8> exprs;
  exprs.reserve(getNumConstraints());

  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getEquality64(i), numDims,
                                              numSyms, localExprs, context));
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getInequality64(i), numDims,
                                              numSyms, localExprs, context));
  return IntegerSet::get(numDims, numSyms, exprs, eqFlags);
}

AffineMap mlir::alignAffineMapWithValues(AffineMap map, ValueRange operands,
                                         ValueRange dims, ValueRange syms,
                                         SmallVector<Value> *newSyms) {
  assert(operands.size() == map.getNumInputs() &&
         "expected same number of operands and map inputs");
  MLIRContext *ctx = map.getContext();
  Builder builder(ctx);
  SmallVector<AffineExpr> dimReplacements(map.getNumDims(), {});
  unsigned numSymbols = syms.size();
  SmallVector<AffineExpr> symReplacements(map.getNumSymbols(), {});
  if (newSyms) {
    newSyms->clear();
    newSyms->append(syms.begin(), syms.end());
  }

  for (const auto &operand : llvm::enumerate(operands)) {
    // Compute replacement dim/sym of operand.
    AffineExpr replacement;
    auto dimIt = std::find(dims.begin(), dims.end(), operand.value());
    auto symIt = std::find(syms.begin(), syms.end(), operand.value());
    if (dimIt != dims.end()) {
      replacement =
          builder.getAffineDimExpr(std::distance(dims.begin(), dimIt));
    } else if (symIt != syms.end()) {
      replacement =
          builder.getAffineSymbolExpr(std::distance(syms.begin(), symIt));
    } else {
      // This operand is neither a dimension nor a symbol. Add it as a new
      // symbol.
      replacement = builder.getAffineSymbolExpr(numSymbols++);
      if (newSyms)
        newSyms->push_back(operand.value());
    }
    // Add to corresponding replacements vector.
    if (operand.index() < map.getNumDims()) {
      dimReplacements[operand.index()] = replacement;
    } else {
      symReplacements[operand.index() - map.getNumDims()] = replacement;
    }
  }

  return map.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                   dims.size(), numSymbols);
}

FlatAffineValueConstraints FlatAffineRelation::getDomainSet() const {
  FlatAffineValueConstraints domain = *this;
  // Convert all range variables to local variables.
  domain.convertToLocal(VarKind::SetDim, getNumDomainDims(),
                        getNumDomainDims() + getNumRangeDims());
  return domain;
}

FlatAffineValueConstraints FlatAffineRelation::getRangeSet() const {
  FlatAffineValueConstraints range = *this;
  // Convert all domain variables to local variables.
  range.convertToLocal(VarKind::SetDim, 0, getNumDomainDims());
  return range;
}

void FlatAffineRelation::compose(const FlatAffineRelation &other) {
  assert(getNumDomainDims() == other.getNumRangeDims() &&
         "Domain of this and range of other do not match");
  assert(std::equal(values.begin(), values.begin() + getNumDomainDims(),
                    other.values.begin() + other.getNumDomainDims()) &&
         "Domain of this and range of other do not match");

  FlatAffineRelation rel = other;

  // Convert `rel` from
  //    [otherDomain] -> [otherRange]
  // to
  //    [otherDomain] -> [otherRange thisRange]
  // and `this` from
  //    [thisDomain] -> [thisRange]
  // to
  //    [otherDomain thisDomain] -> [thisRange].
  unsigned removeDims = rel.getNumRangeDims();
  insertDomainVar(0, rel.getNumDomainDims());
  rel.appendRangeVar(getNumRangeDims());

  // Merge symbol and local variables.
  mergeSymbolVars(rel);
  mergeLocalVars(rel);

  // Convert `rel` from [otherDomain] -> [otherRange thisRange] to
  // [otherDomain] -> [thisRange] by converting first otherRange range vars
  // to local vars.
  rel.convertToLocal(VarKind::SetDim, rel.getNumDomainDims(),
                     rel.getNumDomainDims() + removeDims);
  // Convert `this` from [otherDomain thisDomain] -> [thisRange] to
  // [otherDomain] -> [thisRange] by converting last thisDomain domain vars
  // to local vars.
  convertToLocal(VarKind::SetDim, getNumDomainDims() - removeDims,
                 getNumDomainDims());

  auto thisMaybeValues = getMaybeValues(VarKind::SetDim);
  auto relMaybeValues = rel.getMaybeValues(VarKind::SetDim);

  // Add and match domain of `rel` to domain of `this`.
  for (unsigned i = 0, e = rel.getNumDomainDims(); i < e; ++i)
    if (relMaybeValues[i].has_value())
      setValue(i, *relMaybeValues[i]);
  // Add and match range of `this` to range of `rel`.
  for (unsigned i = 0, e = getNumRangeDims(); i < e; ++i) {
    unsigned rangeIdx = rel.getNumDomainDims() + i;
    if (thisMaybeValues[rangeIdx].has_value())
      rel.setValue(rangeIdx, *thisMaybeValues[rangeIdx]);
  }

  // Append `this` to `rel` and simplify constraints.
  rel.append(*this);
  rel.removeRedundantLocalVars();

  *this = rel;
}

void FlatAffineRelation::inverse() {
  unsigned oldDomain = getNumDomainDims();
  unsigned oldRange = getNumRangeDims();
  // Add new range vars.
  appendRangeVar(oldDomain);
  // Swap new vars with domain.
  for (unsigned i = 0; i < oldDomain; ++i)
    swapVar(i, oldDomain + oldRange + i);
  // Remove the swapped domain.
  removeVarRange(0, oldDomain);
  // Set domain and range as inverse.
  numDomainDims = oldRange;
  numRangeDims = oldDomain;
}

void FlatAffineRelation::insertDomainVar(unsigned pos, unsigned num) {
  assert(pos <= getNumDomainDims() &&
         "Var cannot be inserted at invalid position");
  insertDimVar(pos, num);
  numDomainDims += num;
}

void FlatAffineRelation::insertRangeVar(unsigned pos, unsigned num) {
  assert(pos <= getNumRangeDims() &&
         "Var cannot be inserted at invalid position");
  insertDimVar(getNumDomainDims() + pos, num);
  numRangeDims += num;
}

void FlatAffineRelation::appendDomainVar(unsigned num) {
  insertDimVar(getNumDomainDims(), num);
  numDomainDims += num;
}

void FlatAffineRelation::appendRangeVar(unsigned num) {
  insertDimVar(getNumDimVars(), num);
  numRangeDims += num;
}

void FlatAffineRelation::removeVarRange(VarKind kind, unsigned varStart,
                                        unsigned varLimit) {
  assert(varLimit <= getNumVarKind(kind));
  if (varStart >= varLimit)
    return;

  FlatAffineValueConstraints::removeVarRange(kind, varStart, varLimit);

  // If kind is not SetDim, domain and range don't need to be updated.
  if (kind != VarKind::SetDim)
    return;

  // Compute number of domain and range variables to remove. This is done by
  // intersecting the range of domain/range vars with range of vars to remove.
  unsigned intersectDomainLHS = std::min(varLimit, getNumDomainDims());
  unsigned intersectDomainRHS = varStart;
  unsigned intersectRangeLHS = std::min(varLimit, getNumDimVars());
  unsigned intersectRangeRHS = std::max(varStart, getNumDomainDims());

  if (intersectDomainLHS > intersectDomainRHS)
    numDomainDims -= intersectDomainLHS - intersectDomainRHS;
  if (intersectRangeLHS > intersectRangeRHS)
    numRangeDims -= intersectRangeLHS - intersectRangeRHS;
}

LogicalResult mlir::getRelationFromMap(AffineMap &map,
                                       FlatAffineRelation &rel) {
  // Get flattened affine expressions.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineValueConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localVarCst)))
    return failure();

  unsigned oldDimNum = localVarCst.getNumDimVars();
  unsigned oldCols = localVarCst.getNumCols();
  unsigned numRangeVars = map.getNumResults();
  unsigned numDomainVars = map.getNumDims();

  // Add range as the new expressions.
  localVarCst.appendDimVar(numRangeVars);

  // Add equalities between source and range.
  SmallVector<int64_t, 8> eq(localVarCst.getNumCols());
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0, f = oldDimNum; j < f; ++j)
      eq[j] = flatExprs[i][j];
    for (unsigned j = oldDimNum, f = oldCols; j < f; ++j)
      eq[j + numRangeVars] = flatExprs[i][j];
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[numDomainVars + i] = -1;
    localVarCst.addEquality(eq);
  }

  // Create relation and return success.
  rel = FlatAffineRelation(numDomainVars, numRangeVars, localVarCst);
  return success();
}

LogicalResult mlir::getRelationFromMap(const AffineValueMap &map,
                                       FlatAffineRelation &rel) {

  AffineMap affineMap = map.getAffineMap();
  if (failed(getRelationFromMap(affineMap, rel)))
    return failure();

  // Set symbol values for domain dimensions and symbols.
  for (unsigned i = 0, e = rel.getNumDomainDims(); i < e; ++i)
    rel.setValue(i, map.getOperand(i));
  for (unsigned i = rel.getNumDimVars(), e = rel.getNumDimAndSymbolVars();
       i < e; ++i)
    rel.setValue(i, map.getOperand(i - rel.getNumRangeDims()));

  return success();
}

LogicalResult
mlir::getMultiAffineFunctionFromMap(AffineMap map,
                                    MultiAffineFunction &multiAff) {
  FlatAffineValueConstraints cst;
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  LogicalResult result = getFlattenedAffineExprs(map, &flattenedExprs, &cst);

  if (result.failed())
    return failure();

  DivisionRepr divs = cst.getLocalReprs();
  assert(divs.hasAllReprs() &&
         "AffineMap cannot produce divs without local representation");

  // TODO: We shouldn't have to do this conversion.
  Matrix mat(map.getNumResults(), map.getNumInputs() + divs.getNumDivs() + 1);
  for (unsigned i = 0, e = flattenedExprs.size(); i < e; ++i)
    for (unsigned j = 0, f = flattenedExprs[i].size(); j < f; ++j)
      mat(i, j) = flattenedExprs[i][j];

  multiAff = MultiAffineFunction(
      PresburgerSpace::getRelationSpace(map.getNumDims(), map.getNumResults(),
                                        map.getNumSymbols(), divs.getNumDivs()),
      mat, divs);

  return success();
}
