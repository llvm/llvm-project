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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-structures"

using namespace mlir;
using namespace presburger;

namespace {

// See comments for SimpleAffineExprFlattener.
// An AffineExprFlattener extends a SimpleAffineExprFlattener by recording
// constraint information associated with mod's, floordiv's, and ceildiv's
// in FlatAffineConstraints 'localVarCst'.
struct AffineExprFlattener : public SimpleAffineExprFlattener {
public:
  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  FlatAffineConstraints localVarCst;

  AffineExprFlattener(unsigned nDims, unsigned nSymbols)
      : SimpleAffineExprFlattener(nDims, nSymbols) {
    localVarCst.reset(nDims, nSymbols, /*numLocals=*/0);
  }

private:
  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
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
                        FlatAffineConstraints *localVarCst) {
  if (exprs.empty()) {
    localVarCst->reset(numDims, numSymbols);
    return success();
  }

  AffineExprFlattener flattener(numDims, numSymbols);
  // Use the same flattener to simplify each expression successively. This way
  // local identifiers / expressions are shared.
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
                             FlatAffineConstraints *localVarCst) {
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
    FlatAffineConstraints *localVarCst) {
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
    FlatAffineConstraints *localVarCst) {
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

// Clones this object.
std::unique_ptr<FlatAffineConstraints> FlatAffineConstraints::clone() const {
  return std::make_unique<FlatAffineConstraints>(*this);
}

std::unique_ptr<FlatAffineValueConstraints>
FlatAffineValueConstraints::clone() const {
  return std::make_unique<FlatAffineValueConstraints>(*this);
}

// Construct from an IntegerSet.
FlatAffineConstraints::FlatAffineConstraints(IntegerSet set)
    : IntegerPolyhedron(set.getNumInequalities(), set.getNumEqualities(),
                        set.getNumDims() + set.getNumSymbols() + 1,
                        set.getNumDims(), set.getNumSymbols(),
                        /*numLocals=*/0) {

  // Flatten expressions and add them to the constraint system.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(set, &flatExprs, &localVarCst))) {
    assert(false && "flattening unimplemented for semi-affine integer sets");
    return;
  }
  assert(flatExprs.size() == set.getNumConstraints());
  appendLocalId(/*num=*/localVarCst.getNumLocalIds());

  for (unsigned i = 0, e = flatExprs.size(); i < e; ++i) {
    const auto &flatExpr = flatExprs[i];
    assert(flatExpr.size() == getNumCols());
    if (set.getEqFlags()[i]) {
      addEquality(flatExpr);
    } else {
      addInequality(flatExpr);
    }
  }
  // Add the other constraints involving local id's from flattening.
  append(localVarCst);
}

// Construct from an IntegerSet.
FlatAffineValueConstraints::FlatAffineValueConstraints(IntegerSet set)
    : FlatAffineConstraints(set) {
  values.resize(getNumIds(), None);
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
// All dimensions as set as DimId.
FlatAffineValueConstraints
FlatAffineValueConstraints::getHyperrectangular(ValueRange ivs, ValueRange lbs,
                                                ValueRange ubs) {
  FlatAffineValueConstraints res;
  unsigned nIvs = ivs.size();
  assert(nIvs == lbs.size() && "expected as many lower bounds as ivs");
  assert(nIvs == ubs.size() && "expected as many upper bounds as ivs");

  if (nIvs == 0)
    return res;

  res.appendDimId(ivs);
  unsigned lbsStart = res.appendDimId(lbs);
  unsigned ubsStart = res.appendDimId(ubs);

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

void FlatAffineConstraints::reset(unsigned numReservedInequalities,
                                  unsigned numReservedEqualities,
                                  unsigned newNumReservedCols,
                                  unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  *this = FlatAffineConstraints(numReservedInequalities, numReservedEqualities,
                                newNumReservedCols, newNumDims, newNumSymbols,
                                newNumLocals);
}

void FlatAffineConstraints::reset(unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals) {
  reset(/*numReservedInequalities=*/0, /*numReservedEqualities=*/0,
        /*numReservedCols=*/newNumDims + newNumSymbols + newNumLocals + 1,
        newNumDims, newNumSymbols, newNumLocals);
}

void FlatAffineValueConstraints::reset(unsigned numReservedInequalities,
                                       unsigned numReservedEqualities,
                                       unsigned newNumReservedCols,
                                       unsigned newNumDims,
                                       unsigned newNumSymbols,
                                       unsigned newNumLocals) {
  reset(numReservedInequalities, numReservedEqualities, newNumReservedCols,
        newNumDims, newNumSymbols, newNumLocals, /*valArgs=*/{});
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

unsigned FlatAffineValueConstraints::appendDimId(ValueRange vals) {
  unsigned pos = getNumDimIds();
  insertId(IdKind::SetDim, pos, vals);
  return pos;
}

unsigned FlatAffineValueConstraints::appendSymbolId(ValueRange vals) {
  unsigned pos = getNumSymbolIds();
  insertId(IdKind::Symbol, pos, vals);
  return pos;
}

unsigned FlatAffineValueConstraints::insertDimId(unsigned pos,
                                                 ValueRange vals) {
  return insertId(IdKind::SetDim, pos, vals);
}

unsigned FlatAffineValueConstraints::insertSymbolId(unsigned pos,
                                                    ValueRange vals) {
  return insertId(IdKind::Symbol, pos, vals);
}

unsigned FlatAffineValueConstraints::insertId(IdKind kind, unsigned pos,
                                              unsigned num) {
  unsigned absolutePos = FlatAffineConstraints::insertId(kind, pos, num);
  values.insert(values.begin() + absolutePos, num, None);
  assert(values.size() == getNumIds());
  return absolutePos;
}

unsigned FlatAffineValueConstraints::insertId(IdKind kind, unsigned pos,
                                              ValueRange vals) {
  assert(!vals.empty() && "expected ValueRange with Values");
  unsigned num = vals.size();
  unsigned absolutePos = FlatAffineConstraints::insertId(kind, pos, num);

  // If a Value is provided, insert it; otherwise use None.
  for (unsigned i = 0; i < num; ++i)
    values.insert(values.begin() + absolutePos + i,
                  vals[i] ? Optional<Value>(vals[i]) : None);

  assert(values.size() == getNumIds());
  return absolutePos;
}

bool FlatAffineValueConstraints::hasValues() const {
  return llvm::find_if(values, [](Optional<Value> id) {
           return id.hasValue();
         }) != values.end();
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of identifiers, appearing in the same order.
static bool areIdsAligned(const FlatAffineValueConstraints &a,
                          const FlatAffineValueConstraints &b) {
  return a.getNumDimIds() == b.getNumDimIds() &&
         a.getNumSymbolIds() == b.getNumSymbolIds() &&
         a.getNumIds() == b.getNumIds() &&
         a.getMaybeValues().equals(b.getMaybeValues());
}

/// Calls areIdsAligned to check if two constraint systems have the same set
/// of identifiers in the same order.
bool FlatAffineValueConstraints::areIdsAlignedWithOther(
    const FlatAffineValueConstraints &other) {
  return areIdsAligned(*this, other);
}

/// Checks if the SSA values associated with `cst`'s identifiers in range
/// [start, end) are unique.
static bool LLVM_ATTRIBUTE_UNUSED areIdsUnique(
    const FlatAffineValueConstraints &cst, unsigned start, unsigned end) {

  assert(start <= cst.getNumIds() && "Start position out of bounds");
  assert(end <= cst.getNumIds() && "End position out of bounds");

  if (start >= end)
    return true;

  SmallPtrSet<Value, 8> uniqueIds;
  ArrayRef<Optional<Value>> maybeValues = cst.getMaybeValues();
  for (Optional<Value> val : maybeValues) {
    if (val.hasValue() && !uniqueIds.insert(val.getValue()).second)
      return false;
  }
  return true;
}

/// Checks if the SSA values associated with `cst`'s identifiers are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areIdsUnique(const FlatAffineConstraints &cst) {
  return areIdsUnique(cst, 0, cst.getNumIds());
}

/// Checks if the SSA values associated with `cst`'s identifiers of kind `kind`
/// are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areIdsUnique(const FlatAffineValueConstraints &cst, IdKind kind) {

  if (kind == IdKind::SetDim)
    return areIdsUnique(cst, 0, cst.getNumDimIds());
  if (kind == IdKind::Symbol)
    return areIdsUnique(cst, cst.getNumDimIds(), cst.getNumDimAndSymbolIds());
  if (kind == IdKind::Local)
    return areIdsUnique(cst, cst.getNumDimAndSymbolIds(), cst.getNumIds());
  llvm_unreachable("Unexpected IdKind");
}

/// Merge and align the identifiers of A and B starting at 'offset', so that
/// both constraint systems get the union of the contained identifiers that is
/// dimension-wise and symbol-wise unique; both constraint systems are updated
/// so that they have the union of all identifiers, with A's original
/// identifiers appearing first followed by any of B's identifiers that didn't
/// appear in A. Local identifiers in B that have the same division
/// representation as local identifiers in A are merged into one.
//  E.g.: Input: A has ((%i, %j) [%M, %N]) and B has (%k, %j) [%P, %N, %M])
//        Output: both A, B have (%i, %j, %k) [%M, %N, %P]
static void mergeAndAlignIds(unsigned offset, FlatAffineValueConstraints *a,
                             FlatAffineValueConstraints *b) {
  assert(offset <= a->getNumDimIds() && offset <= b->getNumDimIds());
  // A merge/align isn't meaningful if a cst's ids aren't distinct.
  assert(areIdsUnique(*a) && "A's values aren't unique");
  assert(areIdsUnique(*b) && "B's values aren't unique");

  assert(std::all_of(a->getMaybeValues().begin() + offset,
                     a->getMaybeValues().begin() + a->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  assert(std::all_of(b->getMaybeValues().begin() + offset,
                     b->getMaybeValues().begin() + b->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  SmallVector<Value, 4> aDimValues;
  a->getValues(offset, a->getNumDimIds(), &aDimValues);

  {
    // Merge dims from A into B.
    unsigned d = offset;
    for (auto aDimValue : aDimValues) {
      unsigned loc;
      if (b->findId(aDimValue, &loc)) {
        assert(loc >= offset && "A's dim appears in B's aligned range");
        assert(loc < b->getNumDimIds() &&
               "A's dim appears in B's non-dim position");
        b->swapId(d, loc);
      } else {
        b->insertDimId(d, aDimValue);
      }
      d++;
    }
    // Dimensions that are in B, but not in A, are added at the end.
    for (unsigned t = a->getNumDimIds(), e = b->getNumDimIds(); t < e; t++) {
      a->appendDimId(b->getValue(t));
    }
    assert(a->getNumDimIds() == b->getNumDimIds() &&
           "expected same number of dims");
  }

  // Merge and align symbols of A and B
  a->mergeSymbolIds(*b);
  // Merge and align local ids of A and B
  a->mergeLocalIds(*b);

  assert(areIdsAligned(*a, *b) && "IDs expected to be aligned");
}

// Call 'mergeAndAlignIds' to align constraint systems of 'this' and 'other'.
void FlatAffineValueConstraints::mergeAndAlignIdsWithOther(
    unsigned offset, FlatAffineValueConstraints *other) {
  mergeAndAlignIds(offset, this, other);
}

LogicalResult
FlatAffineValueConstraints::composeMap(const AffineValueMap *vMap) {
  return composeMatchingMap(
      computeAlignedMap(vMap->getAffineMap(), vMap->getOperands()));
}

// Similar to `composeMap` except that no Values need be associated with the
// constraint system nor are they looked at -- the dimensions and symbols of
// `other` are expected to correspond 1:1 to `this` system.
LogicalResult FlatAffineConstraints::composeMatchingMap(AffineMap other) {
  assert(other.getNumDims() == getNumDimIds() && "dim mismatch");
  assert(other.getNumSymbols() == getNumSymbolIds() && "symbol mismatch");

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(flattenAlignedMapAndMergeLocals(other, &flatExprs)))
    return failure();
  assert(flatExprs.size() == other.getNumResults());

  // Add dimensions corresponding to the map's results.
  insertDimId(/*pos=*/0, /*num=*/other.getNumResults());

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
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
    unsigned j = getNumDimIds() + getNumSymbolIds();
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
static void turnSymbolIntoDim(FlatAffineValueConstraints *cst, Value id) {
  unsigned pos;
  if (cst->findId(id, &pos) && pos >= cst->getNumDimIds() &&
      pos < cst->getNumDimAndSymbolIds()) {
    cst->swapId(pos, cst->getNumDimIds());
    cst->setDimSymbolSeparation(cst->getNumSymbolIds() - 1);
  }
}

/// Merge and align symbols of `this` and `other` such that both get union of
/// of symbols that are unique. Symbols in `this` and `other` should be
/// unique. Symbols with Value as `None` are considered to be inequal to all
/// other symbols.
void FlatAffineValueConstraints::mergeSymbolIds(
    FlatAffineValueConstraints &other) {

  assert(areIdsUnique(*this, IdKind::Symbol) && "Symbol ids are not unique");
  assert(areIdsUnique(other, IdKind::Symbol) && "Symbol ids are not unique");

  SmallVector<Value, 4> aSymValues;
  getValues(getNumDimIds(), getNumDimAndSymbolIds(), &aSymValues);

  // Merge symbols: merge symbols into `other` first from `this`.
  unsigned s = other.getNumDimIds();
  for (Value aSymValue : aSymValues) {
    unsigned loc;
    // If the id is a symbol in `other`, then align it, otherwise assume that
    // it is a new symbol
    if (other.findId(aSymValue, &loc) && loc >= other.getNumDimIds() &&
        loc < other.getNumDimAndSymbolIds())
      other.swapId(s, loc);
    else
      other.insertSymbolId(s - other.getNumDimIds(), aSymValue);
    s++;
  }

  // Symbols that are in other, but not in this, are added at the end.
  for (unsigned t = other.getNumDimIds() + getNumSymbolIds(),
                e = other.getNumDimAndSymbolIds();
       t < e; t++)
    insertSymbolId(getNumSymbolIds(), other.getValue(t));

  assert(getNumSymbolIds() == other.getNumSymbolIds() &&
         "expected same number of symbols");
  assert(areIdsUnique(*this, IdKind::Symbol) && "Symbol ids are not unique");
  assert(areIdsUnique(other, IdKind::Symbol) && "Symbol ids are not unique");
}

// Changes all symbol identifiers which are loop IVs to dim identifiers.
void FlatAffineValueConstraints::convertLoopIVSymbolsToDims() {
  // Gather all symbols which are loop IVs.
  SmallVector<Value, 4> loopIVs;
  for (unsigned i = getNumDimIds(), e = getNumDimAndSymbolIds(); i < e; i++) {
    if (hasValue(i) && getForInductionVarOwner(getValue(i)))
      loopIVs.push_back(getValue(i));
  }
  // Turn each symbol in 'loopIVs' into a dim identifier.
  for (auto iv : loopIVs) {
    turnSymbolIntoDim(this, iv);
  }
}

void FlatAffineValueConstraints::addInductionVarOrTerminalSymbol(Value val) {
  if (containsId(val))
    return;

  // Caller is expected to fully compose map/operands if necessary.
  assert((isTopLevelValue(val) || isForInductionVar(val)) &&
         "non-terminal symbol / loop IV expected");
  // Outer loop IVs could be used in forOp's bounds.
  if (auto loop = getForInductionVarOwner(val)) {
    appendDimId(val);
    if (failed(this->addAffineForOpDomain(loop)))
      LLVM_DEBUG(
          loop.emitWarning("failed to add domain info to constraint system"));
    return;
  }
  // Add top level symbol.
  appendSymbolId(val);
  // Check if the symbol is a constant.
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    addBound(BoundType::EQ, val, constOp.value());
}

LogicalResult
FlatAffineValueConstraints::addAffineForOpDomain(AffineForOp forOp) {
  unsigned pos;
  // Pre-condition for this method.
  if (!findId(forOp.getInductionVar(), &pos)) {
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

LogicalResult
FlatAffineValueConstraints::addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                                   ArrayRef<AffineMap> ubMaps,
                                                   ArrayRef<Value> operands) {
  assert(lbMaps.size() == ubMaps.size());
  assert(lbMaps.size() <= getNumDimIds());

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
    // bounds to the system assuming its dimension identifier position is the
    // same as the position of the loop in the loop nest.
    if (lbMap && failed(addBound(BoundType::LB, i, lbMap, operands)))
      return failure();
    if (ubMap && failed(addBound(BoundType::UB, i, ubMap, operands)))
      return failure();
  }
  return success();
}

void FlatAffineValueConstraints::addAffineIfOpDomain(AffineIfOp ifOp) {
  // Create the base constraints from the integer set attached to ifOp.
  FlatAffineValueConstraints cst(ifOp.getIntegerSet());

  // Bind ids in the constraints to ifOp operands.
  SmallVector<Value, 4> operands = ifOp.getOperands();
  cst.setValues(0, cst.getNumDimAndSymbolIds(), operands);

  // Merge the constraints from ifOp to the current domain. We need first merge
  // and align the IDs from both constraints, and then append the constraints
  // from the ifOp into the current one.
  mergeAndAlignIdsWithOther(0, &cst);
  append(cst);
}

bool FlatAffineValueConstraints::hasConsistentState() const {
  return FlatAffineConstraints::hasConsistentState() &&
         values.size() == getNumIds();
}

void FlatAffineValueConstraints::removeIdRange(unsigned idStart,
                                               unsigned idLimit) {
  FlatAffineConstraints::removeIdRange(idStart, idLimit);
  values.erase(values.begin() + idStart, values.begin() + idLimit);
}

// Determine whether the identifier at 'pos' (say id_r) can be expressed as
// modulo of another known identifier (say id_n) w.r.t a constant. For example,
// if the following constraints hold true:
// ```
// 0 <= id_r <= divisor - 1
// id_n - (divisor * q_expr) = id_r
// ```
// where `id_n` is a known identifier (called dividend), and `q_expr` is an
// `AffineExpr` (called the quotient expression), `id_r` can be written as:
//
// `id_r = id_n mod divisor`.
//
// Additionally, in a special case of the above constaints where `q_expr` is an
// identifier itself that is not yet known (say `id_q`), it can be written as a
// floordiv in the following way:
//
// `id_q = id_n floordiv divisor`.
//
// Returns true if the above mod or floordiv are detected, updating 'memo' with
// these new expressions. Returns false otherwise.
static bool detectAsMod(const FlatAffineConstraints &cst, unsigned pos,
                        int64_t lbConst, int64_t ubConst,
                        SmallVectorImpl<AffineExpr> &memo,
                        MLIRContext *context) {
  assert(pos < cst.getNumIds() && "invalid position");

  // Check if a divisor satisfying the condition `0 <= id_r <= divisor - 1` can
  // be determined.
  if (lbConst != 0 || ubConst < 1)
    return false;
  int64_t divisor = ubConst + 1;

  // Check for the aforementioned conditions in each equality.
  for (unsigned curEquality = 0, numEqualities = cst.getNumEqualities();
       curEquality < numEqualities; curEquality++) {
    int64_t coefficientAtPos = cst.atEq(curEquality, pos);
    // If current equality does not involve `id_r`, continue to the next
    // equality.
    if (coefficientAtPos == 0)
      continue;

    // Constant term should be 0 in this equality.
    if (cst.atEq(curEquality, cst.getNumCols() - 1) != 0)
      continue;

    // Traverse through the equality and construct the dividend expression
    // `dividendExpr`, to contain all the identifiers which are known and are
    // not divisible by `(coefficientAtPos * divisor)`. Hope here is that the
    // `dividendExpr` gets simplified into a single identifier `id_n` discussed
    // above.
    auto dividendExpr = getAffineConstantExpr(0, context);

    // Track the terms that go into quotient expression, later used to detect
    // additional floordiv.
    unsigned quotientCount = 0;
    int quotientPosition = -1;
    int quotientSign = 1;

    // Consider each term in the current equality.
    unsigned curId, e;
    for (curId = 0, e = cst.getNumDimAndSymbolIds(); curId < e; ++curId) {
      // Ignore id_r.
      if (curId == pos)
        continue;
      int64_t coefficientOfCurId = cst.atEq(curEquality, curId);
      // Ignore ids that do not contribute to the current equality.
      if (coefficientOfCurId == 0)
        continue;
      // Check if the current id goes into the quotient expression.
      if (coefficientOfCurId % (divisor * coefficientAtPos) == 0) {
        quotientCount++;
        quotientPosition = curId;
        quotientSign = (coefficientOfCurId * coefficientAtPos) > 0 ? 1 : -1;
        continue;
      }
      // Identifiers that are part of dividendExpr should be known.
      if (!memo[curId])
        break;
      // Append the current identifier to the dividend expression.
      dividendExpr = dividendExpr + memo[curId] * coefficientOfCurId;
    }

    // Can't construct expression as it depends on a yet uncomputed id.
    if (curId < e)
      continue;

    // Express `id_r` in terms of the other ids collected so far.
    if (coefficientAtPos > 0)
      dividendExpr = (-dividendExpr).floorDiv(coefficientAtPos);
    else
      dividendExpr = dividendExpr.floorDiv(-coefficientAtPos);

    // Simplify the expression.
    dividendExpr = simplifyAffineExpr(dividendExpr, cst.getNumDimIds(),
                                      cst.getNumSymbolIds());
    // Only if the final dividend expression is just a single id (which we call
    // `id_n`), we can proceed.
    // TODO: Handle AffineSymbolExpr as well. There is no reason to restrict it
    // to dims themselves.
    auto dimExpr = dividendExpr.dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      continue;

    // Express `id_r` as `id_n % divisor` and store the expression in `memo`.
    if (quotientCount >= 1) {
      auto ub = cst.getConstantBound(FlatAffineConstraints::BoundType::UB,
                                     dimExpr.getPosition());
      // If `id_n` has an upperbound that is less than the divisor, mod can be
      // eliminated altogether.
      if (ub.hasValue() && ub.getValue() < divisor)
        memo[pos] = dimExpr;
      else
        memo[pos] = dimExpr % divisor;
      // If a unique quotient `id_q` was seen, it can be expressed as
      // `id_n floordiv divisor`.
      if (quotientCount == 1 && !memo[quotientPosition])
        memo[quotientPosition] = dimExpr.floorDiv(divisor) * quotientSign;

      return true;
    }
  }
  return false;
}

/// Check if the pos^th identifier can be expressed as a floordiv of an affine
/// function of other identifiers (where the divisor is a positive constant)
/// given the initial set of expressions in `exprs`. If it can be, the
/// corresponding position in `exprs` is set as the detected affine expr. For
/// eg: 4q <= i + j <= 4q + 3   <=>   q = (i + j) floordiv 4. An equality can
/// also yield a floordiv: eg.  4q = i + j <=> q = (i + j) floordiv 4. 32q + 28
/// <= i <= 32q + 31 => q = i floordiv 32.
static bool detectAsFloorDiv(const FlatAffineConstraints &cst, unsigned pos,
                             MLIRContext *context,
                             SmallVectorImpl<AffineExpr> &exprs) {
  assert(pos < cst.getNumIds() && "invalid position");

  // Get upper-lower bound pair for this variable.
  SmallVector<bool, 8> foundRepr(cst.getNumIds(), false);
  for (unsigned i = 0, e = cst.getNumIds(); i < e; ++i)
    if (exprs[i])
      foundRepr[i] = true;

  SmallVector<int64_t, 8> dividend;
  unsigned divisor;
  auto ulPair = computeSingleVarRepr(cst, foundRepr, pos, dividend, divisor);

  // No upper-lower bound pair found for this var.
  if (ulPair.kind == ReprKind::None || ulPair.kind == ReprKind::Equality)
    return false;

  // Construct the dividend expression.
  auto dividendExpr = getAffineConstantExpr(dividend.back(), context);
  for (unsigned c = 0, f = cst.getNumIds(); c < f; c++)
    if (dividend[c] != 0)
      dividendExpr = dividendExpr + dividend[c] * exprs[c];

  // Successfully detected the floordiv.
  exprs[pos] = dividendExpr.floorDiv(divisor);
  return true;
}

std::pair<AffineMap, AffineMap> FlatAffineConstraints::getLowerAndUpperBound(
    unsigned pos, unsigned offset, unsigned num, unsigned symStartPos,
    ArrayRef<AffineExpr> localExprs, MLIRContext *context) const {
  assert(pos + offset < getNumDimIds() && "invalid dim start pos");
  assert(symStartPos >= (pos + offset) && "invalid sym start pos");
  assert(getNumLocalIds() == localExprs.size() &&
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
  unsigned symCount = getNumDimAndSymbolIds() - symStartPos;
  lbExprs.reserve(lbIndices.size() + eqIndices.size());
  // Lower bound expressions.
  for (auto idx : lbIndices) {
    auto ineq = getInequality(idx);
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
    auto ineq = getInequality(idx);
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
    auto eq = getEquality(idx);
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
/// identifiers (starting at 'offset') as affine maps of the remaining
/// identifiers (dimensional and symbolic identifiers). Local identifiers are
/// themselves explicitly computed as affine functions of other identifiers in
/// this process if needed.
void FlatAffineConstraints::getSliceBounds(unsigned offset, unsigned num,
                                           MLIRContext *context,
                                           SmallVectorImpl<AffineMap> *lbMaps,
                                           SmallVectorImpl<AffineMap> *ubMaps) {
  assert(num < getNumDimIds() && "invalid range");

  // Basic simplification.
  normalizeConstraintsByGCD();

  LLVM_DEBUG(llvm::dbgs() << "getSliceBounds for first " << num
                          << " identifiers\n");
  LLVM_DEBUG(dump());

  // Record computed/detected identifiers.
  SmallVector<AffineExpr, 8> memo(getNumIds());
  // Initialize dimensional and symbolic identifiers.
  for (unsigned i = 0, e = getNumDimIds(); i < e; i++) {
    if (i < offset)
      memo[i] = getAffineDimExpr(i, context);
    else if (i >= offset + num)
      memo[i] = getAffineDimExpr(i - num, context);
  }
  for (unsigned i = getNumDimIds(), e = getNumDimAndSymbolIds(); i < e; i++)
    memo[i] = getAffineSymbolExpr(i - getNumDimIds(), context);

  bool changed;
  do {
    changed = false;
    // Identify yet unknown identifiers as constants or mod's / floordiv's of
    // other identifiers if possible.
    for (unsigned pos = 0; pos < getNumIds(); pos++) {
      if (memo[pos])
        continue;

      auto lbConst = getConstantBound(BoundType::LB, pos);
      auto ubConst = getConstantBound(BoundType::UB, pos);
      if (lbConst.hasValue() && ubConst.hasValue()) {
        // Detect equality to a constant.
        if (lbConst.getValue() == ubConst.getValue()) {
          memo[pos] = getAffineConstantExpr(lbConst.getValue(), context);
          changed = true;
          continue;
        }

        // Detect an identifier as modulo of another identifier w.r.t a
        // constant.
        if (detectAsMod(*this, pos, lbConst.getValue(), ubConst.getValue(),
                        memo, context)) {
          changed = true;
          continue;
        }
      }

      // Detect an identifier as a floordiv of an affine function of other
      // identifiers (divisor is a positive constant).
      if (detectAsFloorDiv(*this, pos, context, memo)) {
        changed = true;
        continue;
      }

      // Detect an identifier as an expression of other identifiers.
      unsigned idx;
      if (!findConstraintWithNonZeroAt(pos, /*isEq=*/true, &idx)) {
        continue;
      }

      // Build AffineExpr solving for identifier 'pos' in terms of all others.
      auto expr = getAffineConstantExpr(0, context);
      unsigned j, e;
      for (j = 0, e = getNumIds(); j < e; ++j) {
        if (j == pos)
          continue;
        int64_t c = atEq(idx, j);
        if (c == 0)
          continue;
        // If any of the involved IDs hasn't been found yet, we can't proceed.
        if (!memo[j])
          break;
        expr = expr + memo[j] * c;
      }
      if (j < e)
        // Can't construct expression as it depends on a yet uncomputed
        // identifier.
        continue;

      // Add constant term to AffineExpr.
      expr = expr + atEq(idx, getNumIds());
      int64_t vPos = atEq(idx, pos);
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
    // identifier's explicit form is computed (in memo[pos]), it's not updated
    // again.
  } while (changed);

  // Set the lower and upper bound maps for all the identifiers that were
  // computed as affine expressions of the rest as the "detected expr" and
  // "detected expr + 1" respectively; set the undetected ones to null.
  Optional<FlatAffineConstraints> tmpClone;
  for (unsigned pos = 0; pos < num; pos++) {
    unsigned numMapDims = getNumDimIds() - num;
    unsigned numMapSymbols = getNumSymbolIds();
    AffineExpr expr = memo[pos + offset];
    if (expr)
      expr = simplifyAffineExpr(expr, numMapDims, numMapSymbols);

    AffineMap &lbMap = (*lbMaps)[pos];
    AffineMap &ubMap = (*ubMaps)[pos];

    if (expr) {
      lbMap = AffineMap::get(numMapDims, numMapSymbols, expr);
      ubMap = AffineMap::get(numMapDims, numMapSymbols, expr + 1);
    } else {
      // TODO: Whenever there are local identifiers in the dependence
      // constraints, we'll conservatively over-approximate, since we don't
      // always explicitly compute them above (in the while loop).
      if (getNumLocalIds() == 0) {
        // Work on a copy so that we don't update this constraint system.
        if (!tmpClone) {
          tmpClone.emplace(FlatAffineConstraints(*this));
          // Removing redundant inequalities is necessary so that we don't get
          // redundant loop bounds.
          tmpClone->removeRedundantInequalities();
        }
        std::tie(lbMap, ubMap) = tmpClone->getLowerAndUpperBound(
            pos, offset, num, getNumDimIds(), /*localExprs=*/{}, context);
      }

      // If the above fails, we'll just use the constant lower bound and the
      // constant upper bound (if they exist) as the slice bounds.
      // TODO: being conservative for the moment in cases that
      // lead to multiple bounds - until getConstDifference in LoopFusion.cpp is
      // fixed (b/126426796).
      if (!lbMap || lbMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice lb\n");
        auto lbConst = getConstantBound(BoundType::LB, pos + offset);
        if (lbConst.hasValue()) {
          lbMap = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(lbConst.getValue(), context));
        }
      }
      if (!ubMap || ubMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice ub\n");
        auto ubConst = getConstantBound(BoundType::UB, pos + offset);
        if (ubConst.hasValue()) {
          (ubMap) = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(ubConst.getValue() + 1, context));
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

LogicalResult FlatAffineConstraints::flattenAlignedMapAndMergeLocals(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs) {
  FlatAffineConstraints localCst;
  if (failed(getFlattenedAffineExprs(map, flattenedExprs, &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }

  // Add localCst information.
  if (localCst.getNumLocalIds() > 0) {
    unsigned numLocalIds = getNumLocalIds();
    // Insert local dims of localCst at the beginning.
    insertLocalId(/*pos=*/0, /*num=*/localCst.getNumLocalIds());
    // Insert local dims of `this` at the end of localCst.
    localCst.appendLocalId(/*num=*/numLocalIds);
    // Dimensions of localCst and this constraint set match. Append localCst to
    // this constraint set.
    append(localCst);
  }

  return success();
}

LogicalResult FlatAffineConstraints::addBound(BoundType type, unsigned pos,
                                              AffineMap boundMap) {
  assert(boundMap.getNumDims() == getNumDimIds() && "dim mismatch");
  assert(boundMap.getNumSymbols() == getNumSymbolIds() && "symbol mismatch");
  assert(pos < getNumDimAndSymbolIds() && "invalid position");

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
    unsigned j = getNumDimIds() + getNumSymbolIds();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = boundMap.getNumInputs(); i < end; i++, j++) {
      ineq[j] = lower ? -flatExpr[i] : flatExpr[i];
    }
    // Constant term.
    ineq[getNumCols() - 1] =
        lower ? -flatExpr[flatExpr.size() - 1]
              // Upper bound in flattenedExpr is an exclusive one.
              : flatExpr[flatExpr.size() - 1] - 1;
    type == BoundType::EQ ? addEquality(ineq) : addInequality(ineq);
  }

  return success();
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

  dims.reserve(getNumDimIds());
  syms.reserve(getNumSymbolIds());
  for (unsigned i = getIdKindOffset(IdKind::SetDim),
                e = getIdKindEnd(IdKind::SetDim);
       i < e; ++i)
    dims.push_back(values[i] ? *values[i] : Value());
  for (unsigned i = getIdKindOffset(IdKind::Symbol),
                e = getIdKindEnd(IdKind::Symbol);
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
    if (!findId(values[i], &pos))
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

bool FlatAffineValueConstraints::findId(Value val, unsigned *pos) const {
  unsigned i = 0;
  for (const auto &mayBeId : values) {
    if (mayBeId.hasValue() && mayBeId.getValue() == val) {
      *pos = i;
      return true;
    }
    i++;
  }
  return false;
}

bool FlatAffineValueConstraints::containsId(Value val) const {
  return llvm::any_of(values, [&](const Optional<Value> &mayBeId) {
    return mayBeId.hasValue() && mayBeId.getValue() == val;
  });
}

void FlatAffineValueConstraints::swapId(unsigned posA, unsigned posB) {
  FlatAffineConstraints::swapId(posA, posB);
  std::swap(values[posA], values[posB]);
}

void FlatAffineValueConstraints::addBound(BoundType type, Value val,
                                          int64_t value) {
  unsigned pos;
  if (!findId(val, &pos))
    // This is a pre-condition for this method.
    assert(0 && "id not found");
  addBound(type, pos, value);
}

void FlatAffineConstraints::printSpace(raw_ostream &os) const {
  IntegerPolyhedron::printSpace(os);
  os << "(";
  for (unsigned i = 0, e = getNumIds(); i < e; i++) {
    if (auto *valueCstr = dyn_cast<const FlatAffineValueConstraints>(this)) {
      if (valueCstr->hasValue(i))
        os << "Value ";
      else
        os << "None ";
    } else {
      os << "None ";
    }
  }
  os << " const)\n";
}

void FlatAffineConstraints::clearAndCopyFrom(const IntegerPolyhedron &other) {
  if (auto *otherValueSet = dyn_cast<const FlatAffineValueConstraints>(&other))
    assert(!otherValueSet->hasValues() &&
           "cannot copy associated Values into FlatAffineConstraints");

  // Note: Assigment operator does not vtable pointer, so kind does not
  // change.
  if (auto *otherValueSet = dyn_cast<const FlatAffineConstraints>(&other))
    *this = *otherValueSet;
  else
    *static_cast<IntegerPolyhedron *>(this) = other;
}

void FlatAffineValueConstraints::clearAndCopyFrom(
    const IntegerPolyhedron &other) {

  if (auto *otherValueSet =
          dyn_cast<const FlatAffineValueConstraints>(&other)) {
    *this = *otherValueSet;
    return;
  }

  if (auto *otherValueSet = dyn_cast<const FlatAffineValueConstraints>(&other))
    *static_cast<FlatAffineConstraints *>(this) = *otherValueSet;
  else
    *static_cast<IntegerPolyhedron *>(this) = other;

  values.clear();
  values.resize(getNumIds(), None);
}

void FlatAffineValueConstraints::fourierMotzkinEliminate(
    unsigned pos, bool darkShadow, bool *isResultIntegerExact) {
  SmallVector<Optional<Value>, 8> newVals;
  newVals.reserve(getNumIds() - 1);
  newVals.append(values.begin(), values.begin() + pos);
  newVals.append(values.begin() + pos + 1, values.end());
  // Note: Base implementation discards all associated Values.
  FlatAffineConstraints::fourierMotzkinEliminate(pos, darkShadow,
                                                 isResultIntegerExact);
  values = newVals;
  assert(values.size() == getNumIds());
}

void FlatAffineValueConstraints::projectOut(Value val) {
  unsigned pos;
  bool ret = findId(val, &pos);
  assert(ret);
  (void)ret;
  fourierMotzkinEliminate(pos);
}

LogicalResult FlatAffineValueConstraints::unionBoundingBox(
    const FlatAffineValueConstraints &otherCst) {
  assert(otherCst.getNumDimIds() == getNumDimIds() && "dims mismatch");
  assert(otherCst.getMaybeValues()
             .slice(0, getNumDimIds())
             .equals(getMaybeValues().slice(0, getNumDimIds())) &&
         "dim values mismatch");
  assert(otherCst.getNumLocalIds() == 0 && "local ids not supported here");
  assert(getNumLocalIds() == 0 && "local ids not supported yet here");

  // Align `other` to this.
  if (!areIdsAligned(*this, otherCst)) {
    FlatAffineValueConstraints otherCopy(otherCst);
    mergeAndAlignIds(/*offset=*/getNumDimIds(), this, &otherCopy);
    return FlatAffineConstraints::unionBoundingBox(otherCopy);
  }

  return FlatAffineConstraints::unionBoundingBox(otherCst);
}

/// Compute an explicit representation for local vars. For all systems coming
/// from MLIR integer sets, maps, or expressions where local vars were
/// introduced to model floordivs and mods, this always succeeds.
static LogicalResult computeLocalVars(const FlatAffineConstraints &cst,
                                      SmallVectorImpl<AffineExpr> &memo,
                                      MLIRContext *context) {
  unsigned numDims = cst.getNumDimIds();
  unsigned numSyms = cst.getNumSymbolIds();

  // Initialize dimensional and symbolic identifiers.
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
    for (unsigned i = 0, e = cst.getNumLocalIds(); i < e; ++i)
      if (!memo[numDims + numSyms + i] &&
          detectAsFloorDiv(cst, /*pos=*/numDims + numSyms + i, context, memo))
        changed = true;
  } while (changed);

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(cst.getNumLocalIds());
  return success(
      llvm::all_of(localExprs, [](AffineExpr expr) { return expr; }));
}

void FlatAffineValueConstraints::getIneqAsAffineValueMap(
    unsigned pos, unsigned ineqPos, AffineValueMap &vmap,
    MLIRContext *context) const {
  unsigned numDims = getNumDimIds();
  unsigned numSyms = getNumSymbolIds();

  assert(pos < numDims && "invalid position");
  assert(ineqPos < getNumInequalities() && "invalid inequality position");

  // Get expressions for local vars.
  SmallVector<AffineExpr, 8> memo(getNumIds(), AffineExpr());
  if (failed(computeLocalVars(*this, memo, context)))
    assert(false &&
           "one or more local exprs do not have an explicit representation");
  auto localExprs = ArrayRef<AffineExpr>(memo).take_back(getNumLocalIds());

  // Compute the AffineExpr lower/upper bound for this inequality.
  ArrayRef<int64_t> inequality = getInequality(ineqPos);
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
  getValues(pos + 1, getNumDimAndSymbolIds(), &trailingOperands);
  operands.append(trailingOperands.begin(), trailingOperands.end());
  vmap.reset(AffineMap::get(numDims - 1, numSyms, boundExpr), operands);
}

IntegerSet FlatAffineConstraints::getAsIntegerSet(MLIRContext *context) const {
  if (getNumConstraints() == 0)
    // Return universal set (always true): 0 == 0.
    return IntegerSet::get(getNumDimIds(), getNumSymbolIds(),
                           getAffineConstantExpr(/*constant=*/0, context),
                           /*eqFlags=*/true);

  // Construct local references.
  SmallVector<AffineExpr, 8> memo(getNumIds(), AffineExpr());

  if (failed(computeLocalVars(*this, memo, context))) {
    // Check if the local variables without an explicit representation have
    // zero coefficients everywhere.
    SmallVector<unsigned> noLocalRepVars;
    unsigned numDimsSymbols = getNumDimAndSymbolIds();
    for (unsigned i = numDimsSymbols, e = getNumIds(); i < e; ++i) {
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
      ArrayRef<AffineExpr>(memo).take_back(getNumLocalIds());

  // Construct the IntegerSet from the equalities/inequalities.
  unsigned numDims = getNumDimIds();
  unsigned numSyms = getNumSymbolIds();

  SmallVector<bool, 16> eqFlags(getNumConstraints());
  std::fill(eqFlags.begin(), eqFlags.begin() + getNumEqualities(), true);
  std::fill(eqFlags.begin() + getNumEqualities(), eqFlags.end(), false);

  SmallVector<AffineExpr, 8> exprs;
  exprs.reserve(getNumConstraints());

  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getEquality(i), numDims, numSyms,
                                              localExprs, context));
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getInequality(i), numDims,
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
  domain.convertDimToLocal(getNumDomainDims(),
                           getNumDomainDims() + getNumRangeDims());
  return domain;
}

FlatAffineValueConstraints FlatAffineRelation::getRangeSet() const {
  FlatAffineValueConstraints range = *this;
  // Convert all domain variables to local variables.
  range.convertDimToLocal(0, getNumDomainDims());
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
  insertDomainId(0, rel.getNumDomainDims());
  rel.appendRangeId(getNumRangeDims());

  // Merge symbol and local identifiers.
  mergeSymbolIds(rel);
  mergeLocalIds(rel);

  // Convert `rel` from [otherDomain] -> [otherRange thisRange] to
  // [otherDomain] -> [thisRange] by converting first otherRange range ids
  // to local ids.
  rel.convertDimToLocal(rel.getNumDomainDims(),
                        rel.getNumDomainDims() + removeDims);
  // Convert `this` from [otherDomain thisDomain] -> [thisRange] to
  // [otherDomain] -> [thisRange] by converting last thisDomain domain ids
  // to local ids.
  convertDimToLocal(getNumDomainDims() - removeDims, getNumDomainDims());

  auto thisMaybeValues = getMaybeDimValues();
  auto relMaybeValues = rel.getMaybeDimValues();

  // Add and match domain of `rel` to domain of `this`.
  for (unsigned i = 0, e = rel.getNumDomainDims(); i < e; ++i)
    if (relMaybeValues[i].hasValue())
      setValue(i, relMaybeValues[i].getValue());
  // Add and match range of `this` to range of `rel`.
  for (unsigned i = 0, e = getNumRangeDims(); i < e; ++i) {
    unsigned rangeIdx = rel.getNumDomainDims() + i;
    if (thisMaybeValues[rangeIdx].hasValue())
      rel.setValue(rangeIdx, thisMaybeValues[rangeIdx].getValue());
  }

  // Append `this` to `rel` and simplify constraints.
  rel.append(*this);
  rel.removeRedundantLocalVars();

  *this = rel;
}

void FlatAffineRelation::inverse() {
  unsigned oldDomain = getNumDomainDims();
  unsigned oldRange = getNumRangeDims();
  // Add new range ids.
  appendRangeId(oldDomain);
  // Swap new ids with domain.
  for (unsigned i = 0; i < oldDomain; ++i)
    swapId(i, oldDomain + oldRange + i);
  // Remove the swapped domain.
  removeIdRange(0, oldDomain);
  // Set domain and range as inverse.
  numDomainDims = oldRange;
  numRangeDims = oldDomain;
}

void FlatAffineRelation::insertDomainId(unsigned pos, unsigned num) {
  assert(pos <= getNumDomainDims() &&
         "Id cannot be inserted at invalid position");
  insertDimId(pos, num);
  numDomainDims += num;
}

void FlatAffineRelation::insertRangeId(unsigned pos, unsigned num) {
  assert(pos <= getNumRangeDims() &&
         "Id cannot be inserted at invalid position");
  insertDimId(getNumDomainDims() + pos, num);
  numRangeDims += num;
}

void FlatAffineRelation::appendDomainId(unsigned num) {
  insertDimId(getNumDomainDims(), num);
  numDomainDims += num;
}

void FlatAffineRelation::appendRangeId(unsigned num) {
  insertDimId(getNumDimIds(), num);
  numRangeDims += num;
}

void FlatAffineRelation::removeIdRange(unsigned idStart, unsigned idLimit) {
  if (idStart >= idLimit)
    return;

  // Compute number of domain and range identifiers to remove. This is done by
  // intersecting the range of domain/range ids with range of ids to remove.
  unsigned intersectDomainLHS = std::min(idLimit, getNumDomainDims());
  unsigned intersectDomainRHS = idStart;
  unsigned intersectRangeLHS = std::min(idLimit, getNumDimIds());
  unsigned intersectRangeRHS = std::max(idStart, getNumDomainDims());

  FlatAffineValueConstraints::removeIdRange(idStart, idLimit);

  if (intersectDomainLHS > intersectDomainRHS)
    numDomainDims -= intersectDomainLHS - intersectDomainRHS;
  if (intersectRangeLHS > intersectRangeRHS)
    numRangeDims -= intersectRangeLHS - intersectRangeRHS;
}

LogicalResult mlir::getRelationFromMap(AffineMap &map,
                                       FlatAffineRelation &rel) {
  // Get flattened affine expressions.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localVarCst)))
    return failure();

  unsigned oldDimNum = localVarCst.getNumDimIds();
  unsigned oldCols = localVarCst.getNumCols();
  unsigned numRangeIds = map.getNumResults();
  unsigned numDomainIds = map.getNumDims();

  // Add range as the new expressions.
  localVarCst.appendDimId(numRangeIds);

  // Add equalities between source and range.
  SmallVector<int64_t, 8> eq(localVarCst.getNumCols());
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0, f = oldDimNum; j < f; ++j)
      eq[j] = flatExprs[i][j];
    for (unsigned j = oldDimNum, f = oldCols; j < f; ++j)
      eq[j + numRangeIds] = flatExprs[i][j];
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[numDomainIds + i] = -1;
    localVarCst.addEquality(eq);
  }

  // Create relation and return success.
  rel = FlatAffineRelation(numDomainIds, numRangeIds, localVarCst);
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
  for (unsigned i = rel.getNumDimIds(), e = rel.getNumDimAndSymbolIds(); i < e;
       ++i)
    rel.setValue(i, map.getOperand(i - rel.getNumRangeDims()));

  return success();
}
