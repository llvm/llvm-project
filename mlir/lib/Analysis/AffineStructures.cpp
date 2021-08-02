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

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LinearTransform.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
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
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

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

  AffineExprFlattener(unsigned nDims, unsigned nSymbols, MLIRContext *ctx)
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

} // end anonymous namespace

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

  AffineExprFlattener flattener(numDims, numSymbols, exprs[0].getContext());
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
// FlatAffineConstraints.
//===----------------------------------------------------------------------===//

// Clones this object.
std::unique_ptr<FlatAffineConstraints> FlatAffineConstraints::clone() const {
  return std::make_unique<FlatAffineConstraints>(*this);
}

// Construct from an IntegerSet.
FlatAffineConstraints::FlatAffineConstraints(IntegerSet set)
    : numIds(set.getNumDims() + set.getNumSymbols()), numDims(set.getNumDims()),
      numSymbols(set.getNumSymbols()),
      equalities(0, numIds + 1, set.getNumEqualities(), numIds + 1),
      inequalities(0, numIds + 1, set.getNumInequalities(), numIds + 1) {
  ids.resize(numIds, None);

  // Flatten expressions and add them to the constraint system.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(set, &flatExprs, &localVarCst))) {
    assert(false && "flattening unimplemented for semi-affine integer sets");
    return;
  }
  assert(flatExprs.size() == set.getNumConstraints());
  for (unsigned l = 0, e = localVarCst.getNumLocalIds(); l < e; l++) {
    addLocalId(getNumLocalIds());
  }

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

void FlatAffineConstraints::reset(unsigned numReservedInequalities,
                                  unsigned numReservedEqualities,
                                  unsigned newNumReservedCols,
                                  unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<Value> idArgs) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  SmallVector<Optional<Value>, 8> newIds;
  if (!idArgs.empty())
    newIds.assign(idArgs.begin(), idArgs.end());

  *this = FlatAffineConstraints(numReservedInequalities, numReservedEqualities,
                                newNumReservedCols, newNumDims, newNumSymbols,
                                newNumLocals, newIds);
}

void FlatAffineConstraints::reset(unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<Value> idArgs) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals, idArgs);
}

void FlatAffineConstraints::append(const FlatAffineConstraints &other) {
  assert(other.getNumCols() == getNumCols());
  assert(other.getNumDimIds() == getNumDimIds());
  assert(other.getNumSymbolIds() == getNumSymbolIds());

  inequalities.reserveRows(inequalities.getNumRows() +
                           other.getNumInequalities());
  equalities.reserveRows(equalities.getNumRows() + other.getNumEqualities());

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

void FlatAffineConstraints::addLocalId(unsigned pos) {
  addId(IdKind::Local, pos);
}

void FlatAffineConstraints::addDimId(unsigned pos, Value id) {
  addId(IdKind::Dimension, pos, id);
}

void FlatAffineConstraints::addSymbolId(unsigned pos, Value id) {
  addId(IdKind::Symbol, pos, id);
}

/// Adds a dimensional identifier. The added column is initialized to
/// zero.
void FlatAffineConstraints::addId(IdKind kind, unsigned pos, Value id) {
  if (kind == IdKind::Dimension)
    assert(pos <= getNumDimIds());
  else if (kind == IdKind::Symbol)
    assert(pos <= getNumSymbolIds());
  else
    assert(pos <= getNumLocalIds());

  int absolutePos;
  if (kind == IdKind::Dimension) {
    absolutePos = pos;
    numDims++;
  } else if (kind == IdKind::Symbol) {
    absolutePos = pos + getNumDimIds();
    numSymbols++;
  } else {
    absolutePos = pos + getNumDimIds() + getNumSymbolIds();
  }
  numIds++;

  inequalities.insertColumn(absolutePos);
  equalities.insertColumn(absolutePos);

  // If an 'id' is provided, insert it; otherwise use None.
  if (id)
    ids.insert(ids.begin() + absolutePos, id);
  else
    ids.insert(ids.begin() + absolutePos, None);
  assert(ids.size() == getNumIds());
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of identifiers, appearing in the same order.
static bool areIdsAligned(const FlatAffineConstraints &a,
                          const FlatAffineConstraints &b) {
  return a.getNumDimIds() == b.getNumDimIds() &&
         a.getNumSymbolIds() == b.getNumSymbolIds() &&
         a.getNumIds() == b.getNumIds() && a.getIds().equals(b.getIds());
}

/// Calls areIdsAligned to check if two constraint systems have the same set
/// of identifiers in the same order.
bool FlatAffineConstraints::areIdsAlignedWithOther(
    const FlatAffineConstraints &other) {
  return areIdsAligned(*this, other);
}

/// Checks if the SSA values associated with `cst''s identifiers are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areIdsUnique(const FlatAffineConstraints &cst) {
  SmallPtrSet<Value, 8> uniqueIds;
  for (auto id : cst.getIds()) {
    if (id.hasValue() && !uniqueIds.insert(id.getValue()).second)
      return false;
  }
  return true;
}

/// Merge and align the identifiers of A and B starting at 'offset', so that
/// both constraint systems get the union of the contained identifiers that is
/// dimension-wise and symbol-wise unique; both constraint systems are updated
/// so that they have the union of all identifiers, with A's original
/// identifiers appearing first followed by any of B's identifiers that didn't
/// appear in A. Local identifiers of each system are by design separate/local
/// and are placed one after other (A's followed by B's).
//  Eg: Input: A has ((%i %j) [%M %N]) and B has (%k, %j) [%P, %N, %M])
//      Output: both A, B have (%i, %j, %k) [%M, %N, %P]
//
static void mergeAndAlignIds(unsigned offset, FlatAffineConstraints *a,
                             FlatAffineConstraints *b) {
  assert(offset <= a->getNumDimIds() && offset <= b->getNumDimIds());
  // A merge/align isn't meaningful if a cst's ids aren't distinct.
  assert(areIdsUnique(*a) && "A's id values aren't unique");
  assert(areIdsUnique(*b) && "B's id values aren't unique");

  assert(std::all_of(a->getIds().begin() + offset,
                     a->getIds().begin() + a->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  assert(std::all_of(b->getIds().begin() + offset,
                     b->getIds().begin() + b->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  // Place local id's of A after local id's of B.
  for (unsigned l = 0, e = a->getNumLocalIds(); l < e; l++) {
    b->addLocalId(0);
  }
  for (unsigned t = 0, e = b->getNumLocalIds() - a->getNumLocalIds(); t < e;
       t++) {
    a->addLocalId(a->getNumLocalIds());
  }

  SmallVector<Value, 4> aDimValues, aSymValues;
  a->getIdValues(offset, a->getNumDimIds(), &aDimValues);
  a->getIdValues(a->getNumDimIds(), a->getNumDimAndSymbolIds(), &aSymValues);
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
        b->addDimId(d);
        b->setIdValue(d, aDimValue);
      }
      d++;
    }

    // Dimensions that are in B, but not in A, are added at the end.
    for (unsigned t = a->getNumDimIds(), e = b->getNumDimIds(); t < e; t++) {
      a->addDimId(a->getNumDimIds());
      a->setIdValue(a->getNumDimIds() - 1, b->getIdValue(t));
    }
  }
  {
    // Merge symbols: merge A's symbols into B first.
    unsigned s = b->getNumDimIds();
    for (auto aSymValue : aSymValues) {
      unsigned loc;
      if (b->findId(aSymValue, &loc)) {
        assert(loc >= b->getNumDimIds() && loc < b->getNumDimAndSymbolIds() &&
               "A's symbol appears in B's non-symbol position");
        b->swapId(s, loc);
      } else {
        b->addSymbolId(s - b->getNumDimIds());
        b->setIdValue(s, aSymValue);
      }
      s++;
    }
    // Symbols that are in B, but not in A, are added at the end.
    for (unsigned t = a->getNumDimAndSymbolIds(),
                  e = b->getNumDimAndSymbolIds();
         t < e; t++) {
      a->addSymbolId(a->getNumSymbolIds());
      a->setIdValue(a->getNumDimAndSymbolIds() - 1, b->getIdValue(t));
    }
  }
  assert(areIdsAligned(*a, *b) && "IDs expected to be aligned");
}

// Call 'mergeAndAlignIds' to align constraint systems of 'this' and 'other'.
void FlatAffineConstraints::mergeAndAlignIdsWithOther(
    unsigned offset, FlatAffineConstraints *other) {
  mergeAndAlignIds(offset, this, other);
}

// This routine may add additional local variables if the flattened expression
// corresponding to the map has such variables due to mod's, ceildiv's, and
// floordiv's in it.
LogicalResult FlatAffineConstraints::composeMap(const AffineValueMap *vMap) {
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localCst;
  if (failed(getFlattenedAffineExprs(vMap->getAffineMap(), &flatExprs,
                                     &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }
  assert(flatExprs.size() == vMap->getNumResults());

  // Add localCst information.
  if (localCst.getNumLocalIds() > 0) {
    localCst.setIdValues(0, /*end=*/localCst.getNumDimAndSymbolIds(),
                         /*values=*/vMap->getOperands());
    // Align localCst and this.
    mergeAndAlignIds(/*offset=*/0, &localCst, this);
    // Finally, append localCst to this constraint set.
    append(localCst);
  }

  // Add dimensions corresponding to the map's results.
  for (unsigned t = 0, e = vMap->getNumResults(); t < e; t++) {
    // TODO: Consider using a batched version to add a range of IDs.
    addDimId(0);
  }

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
  // For eg: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  //  d_r - 16*i0 - i1 = 0. Hence, when flattening say (i0 + 1, i0 + 8*i2), we
  //  add two equalities overall: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= vMap->getNumOperands() + 1);

    // eqToAdd is the equality corresponding to the flattened affine expression.
    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, e = vMap->getNumOperands(); i < e; i++) {
      unsigned loc;
      bool ret = findId(vMap->getOperand(i), &loc);
      assert(ret && "value map's id can't be found");
      (void)ret;
      // Negate 'eq[r]' since the newly added dimension will be set to this one.
      eqToAdd[loc] = -flatExpr[i];
    }
    // Local vars common to eq and localCst are at the beginning.
    unsigned j = getNumDimIds() + getNumSymbolIds();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = vMap->getNumOperands(); i < end; i++, j++) {
      eqToAdd[j] = -flatExpr[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -flatExpr[flatExpr.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }

  return success();
}

// Similar to composeMap except that no Value's need be associated with the
// constraint system nor are they looked at -- since the dimensions and
// symbols of 'other' are expected to correspond 1:1 to 'this' system. It
// is thus not convenient to share code with composeMap.
LogicalResult FlatAffineConstraints::composeMatchingMap(AffineMap other) {
  assert(other.getNumDims() == getNumDimIds() && "dim mismatch");
  assert(other.getNumSymbols() == getNumSymbolIds() && "symbol mismatch");

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localCst;
  if (failed(getFlattenedAffineExprs(other, &flatExprs, &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }
  assert(flatExprs.size() == other.getNumResults());

  // Add localCst information.
  if (localCst.getNumLocalIds() > 0) {
    // Place local id's of A after local id's of B.
    for (unsigned l = 0, e = localCst.getNumLocalIds(); l < e; l++) {
      addLocalId(0);
    }
    // Finally, append localCst to this constraint set.
    append(localCst);
  }

  // Add dimensions corresponding to the map's results.
  for (unsigned t = 0, e = other.getNumResults(); t < e; t++) {
    addDimId(0);
  }

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
  // For eg: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  //  d_r - 16*i0 - i1 = 0. Hence, when flattening say (i0 + 1, i0 + 8*i2), we
  //  add two equalities overall: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= other.getNumInputs() + 1);

    // eqToAdd is the equality corresponding to the flattened affine expression.
    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, f = other.getNumInputs(); i < f; i++) {
      // Negate 'eq[r]' since the newly added dimension will be set to this one.
      eqToAdd[e + i] = -flatExpr[i];
    }
    // Local vars common to eq and localCst are at the beginning.
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

// Turn a dimension into a symbol.
static void turnDimIntoSymbol(FlatAffineConstraints *cst, Value id) {
  unsigned pos;
  if (cst->findId(id, &pos) && pos < cst->getNumDimIds()) {
    cst->swapId(pos, cst->getNumDimIds() - 1);
    cst->setDimSymbolSeparation(cst->getNumSymbolIds() + 1);
  }
}

// Turn a symbol into a dimension.
static void turnSymbolIntoDim(FlatAffineConstraints *cst, Value id) {
  unsigned pos;
  if (cst->findId(id, &pos) && pos >= cst->getNumDimIds() &&
      pos < cst->getNumDimAndSymbolIds()) {
    cst->swapId(pos, cst->getNumDimIds());
    cst->setDimSymbolSeparation(cst->getNumSymbolIds() - 1);
  }
}

// Changes all symbol identifiers which are loop IVs to dim identifiers.
void FlatAffineConstraints::convertLoopIVSymbolsToDims() {
  // Gather all symbols which are loop IVs.
  SmallVector<Value, 4> loopIVs;
  for (unsigned i = getNumDimIds(), e = getNumDimAndSymbolIds(); i < e; i++) {
    if (ids[i].hasValue() && getForInductionVarOwner(ids[i].getValue()))
      loopIVs.push_back(ids[i].getValue());
  }
  // Turn each symbol in 'loopIVs' into a dim identifier.
  for (auto iv : loopIVs) {
    turnSymbolIntoDim(this, iv);
  }
}

void FlatAffineConstraints::addInductionVarOrTerminalSymbol(Value id) {
  if (containsId(id))
    return;

  // Caller is expected to fully compose map/operands if necessary.
  assert((isTopLevelValue(id) || isForInductionVar(id)) &&
         "non-terminal symbol / loop IV expected");
  // Outer loop IVs could be used in forOp's bounds.
  if (auto loop = getForInductionVarOwner(id)) {
    addDimId(getNumDimIds(), id);
    if (failed(this->addAffineForOpDomain(loop)))
      LLVM_DEBUG(
          loop.emitWarning("failed to add domain info to constraint system"));
    return;
  }
  // Add top level symbol.
  addSymbolId(getNumSymbolIds(), id);
  // Check if the symbol is a constant.
  if (auto constOp = id.getDefiningOp<ConstantIndexOp>())
    setIdToConstant(id, constOp.getValue());
}

LogicalResult FlatAffineConstraints::addAffineForOpDomain(AffineForOp forOp) {
  unsigned pos;
  // Pre-condition for this method.
  if (!findId(forOp.getInductionVar(), &pos)) {
    assert(false && "Value not found");
    return failure();
  }

  int64_t step = forOp.getStep();
  if (step != 1) {
    if (!forOp.hasConstantLowerBound())
      forOp.emitWarning("domain conservatively approximated");
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
    addConstantLowerBound(pos, forOp.getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    if (failed(addLowerOrUpperBound(pos, forOp.getLowerBoundMap(),
                                    forOp.getLowerBoundOperands(),
                                    /*eq=*/false, /*lower=*/true)))
      return failure();
  }

  if (forOp.hasConstantUpperBound()) {
    addConstantUpperBound(pos, forOp.getConstantUpperBound() - 1);
    return success();
  }
  // Non-constant upper bound case.
  return addLowerOrUpperBound(pos, forOp.getUpperBoundMap(),
                              forOp.getUpperBoundOperands(),
                              /*eq=*/false, /*lower=*/false);
}

/// Adds constraints (lower and upper bounds) for each loop in the loop nest
/// described by the bound maps 'lbMaps' and 'ubMaps' of a computation slice.
/// Every pair ('lbMaps[i]', 'ubMaps[i]') describes the bounds of a loop in
/// the nest, sorted outer-to-inner. 'operands' contains the bound operands
/// for a single bound map. All the bound maps will use the same bound
/// operands. Note that some loops described by a computation slice might not
/// exist yet in the IR so the Value attached to those dimension identifiers
/// might be empty. For that reason, this method doesn't perform Value
/// look-ups to retrieve the dimension identifier positions. Instead, it
/// assumes the position of the dim identifiers in the constraint system is
/// the same as the position of the loop in the loop nest.
LogicalResult
FlatAffineConstraints::addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
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
    if (lbMap && failed(addLowerOrUpperBound(i, lbMap, operands, /*eq=*/false,
                                             /*lower=*/true)))
      return failure();

    if (ubMap && failed(addLowerOrUpperBound(i, ubMap, operands, /*eq=*/false,
                                             /*lower=*/false)))
      return failure();
  }
  return success();
}

void FlatAffineConstraints::addAffineIfOpDomain(AffineIfOp ifOp) {
  // Create the base constraints from the integer set attached to ifOp.
  FlatAffineConstraints cst(ifOp.getIntegerSet());

  // Bind ids in the constraints to ifOp operands.
  SmallVector<Value, 4> operands = ifOp.getOperands();
  cst.setIdValues(0, cst.getNumDimAndSymbolIds(), operands);

  // Merge the constraints from ifOp to the current domain. We need first merge
  // and align the IDs from both constraints, and then append the constraints
  // from the ifOp into the current one.
  mergeAndAlignIdsWithOther(0, &cst);
  append(cst);
}

// Searches for a constraint with a non-zero coefficient at 'colIdx' in
// equality (isEq=true) or inequality (isEq=false) constraints.
// Returns true and sets row found in search in 'rowIdx'.
// Returns false otherwise.
static bool findConstraintWithNonZeroAt(const FlatAffineConstraints &cst,
                                        unsigned colIdx, bool isEq,
                                        unsigned *rowIdx) {
  assert(colIdx < cst.getNumCols() && "position out of bounds");
  auto at = [&](unsigned rowIdx) -> int64_t {
    return isEq ? cst.atEq(rowIdx, colIdx) : cst.atIneq(rowIdx, colIdx);
  };
  unsigned e = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  for (*rowIdx = 0; *rowIdx < e; ++(*rowIdx)) {
    if (at(*rowIdx) != 0) {
      return true;
    }
  }
  return false;
}

// Normalizes the coefficient values across all columns in 'rowIDx' by their
// GCD in equality or inequality constraints as specified by 'isEq'.
template <bool isEq>
static void normalizeConstraintByGCD(FlatAffineConstraints *constraints,
                                     unsigned rowIdx) {
  auto at = [&](unsigned colIdx) -> int64_t {
    return isEq ? constraints->atEq(rowIdx, colIdx)
                : constraints->atIneq(rowIdx, colIdx);
  };
  uint64_t gcd = std::abs(at(0));
  for (unsigned j = 1, e = constraints->getNumCols(); j < e; ++j) {
    gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(at(j)));
  }
  if (gcd > 0 && gcd != 1) {
    for (unsigned j = 0, e = constraints->getNumCols(); j < e; ++j) {
      int64_t v = at(j) / static_cast<int64_t>(gcd);
      isEq ? constraints->atEq(rowIdx, j) = v
           : constraints->atIneq(rowIdx, j) = v;
    }
  }
}

void FlatAffineConstraints::normalizeConstraintsByGCD() {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    normalizeConstraintByGCD</*isEq=*/true>(this, i);
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    normalizeConstraintByGCD</*isEq=*/false>(this, i);
  }
}

bool FlatAffineConstraints::hasConsistentState() const {
  if (!inequalities.hasConsistentState())
    return false;
  if (!equalities.hasConsistentState())
    return false;
  if (ids.size() != getNumIds())
    return false;

  // Catches errors where numDims, numSymbols, numIds aren't consistent.
  if (numDims > numIds || numSymbols > numIds || numDims + numSymbols > numIds)
    return false;

  return true;
}

/// Checks all rows of equality/inequality constraints for trivial
/// contradictions (for example: 1 == 0, 0 >= 1), which may have surfaced
/// after elimination. Returns 'true' if an invalid constraint is found;
/// 'false' otherwise.
bool FlatAffineConstraints::hasInvalidConstraint() const {
  assert(hasConsistentState());
  auto check = [&](bool isEq) -> bool {
    unsigned numCols = getNumCols();
    unsigned numRows = isEq ? getNumEqualities() : getNumInequalities();
    for (unsigned i = 0, e = numRows; i < e; ++i) {
      unsigned j;
      for (j = 0; j < numCols - 1; ++j) {
        int64_t v = isEq ? atEq(i, j) : atIneq(i, j);
        // Skip rows with non-zero variable coefficients.
        if (v != 0)
          break;
      }
      if (j < numCols - 1) {
        continue;
      }
      // Check validity of constant term at 'numCols - 1' w.r.t 'isEq'.
      // Example invalid constraints include: '1 == 0' or '-1 >= 0'
      int64_t v = isEq ? atEq(i, numCols - 1) : atIneq(i, numCols - 1);
      if ((isEq && v != 0) || (!isEq && v < 0)) {
        return true;
      }
    }
    return false;
  };
  if (check(/*isEq=*/true))
    return true;
  return check(/*isEq=*/false);
}

// Eliminate identifier from constraint at 'rowIdx' based on coefficient at
// pivotRow, pivotCol. Columns in range [elimColStart, pivotCol) will not be
// updated as they have already been eliminated.
static void eliminateFromConstraint(FlatAffineConstraints *constraints,
                                    unsigned rowIdx, unsigned pivotRow,
                                    unsigned pivotCol, unsigned elimColStart,
                                    bool isEq) {
  // Skip if equality 'rowIdx' if same as 'pivotRow'.
  if (isEq && rowIdx == pivotRow)
    return;
  auto at = [&](unsigned i, unsigned j) -> int64_t {
    return isEq ? constraints->atEq(i, j) : constraints->atIneq(i, j);
  };
  int64_t leadCoeff = at(rowIdx, pivotCol);
  // Skip if leading coefficient at 'rowIdx' is already zero.
  if (leadCoeff == 0)
    return;
  int64_t pivotCoeff = constraints->atEq(pivotRow, pivotCol);
  int64_t sign = (leadCoeff * pivotCoeff > 0) ? -1 : 1;
  int64_t lcm = mlir::lcm(pivotCoeff, leadCoeff);
  int64_t pivotMultiplier = sign * (lcm / std::abs(pivotCoeff));
  int64_t rowMultiplier = lcm / std::abs(leadCoeff);

  unsigned numCols = constraints->getNumCols();
  for (unsigned j = 0; j < numCols; ++j) {
    // Skip updating column 'j' if it was just eliminated.
    if (j >= elimColStart && j < pivotCol)
      continue;
    int64_t v = pivotMultiplier * constraints->atEq(pivotRow, j) +
                rowMultiplier * at(rowIdx, j);
    isEq ? constraints->atEq(rowIdx, j) = v
         : constraints->atIneq(rowIdx, j) = v;
  }
}

// Removes identifiers in column range [idStart, idLimit), and copies any
// remaining valid data into place, and updates member variables.
void FlatAffineConstraints::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit < getNumCols() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < numIds && "invalid idStart position");

  // TODO: Make 'removeIdRange' a lambda called from here.
  // Remove eliminated identifiers from the constraints..
  equalities.removeColumns(idStart, idLimit - idStart);
  inequalities.removeColumns(idStart, idLimit - idStart);

  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  unsigned numLocalsEliminated = 0;
  unsigned numColsEliminated = idLimit - idStart;
  if (idStart < numDims) {
    numDimsEliminated = std::min(numDims, idLimit) - idStart;
  }
  // Check how many local id's were removed. Note that our identifier order is
  // [dims, symbols, locals]. Local id start at position numDims + numSymbols.
  if (idLimit > numDims + numSymbols) {
    numLocalsEliminated = std::min(
        idLimit - std::max(idStart, numDims + numSymbols), getNumLocalIds());
  }
  unsigned numSymbolsEliminated =
      numColsEliminated - numDimsEliminated - numLocalsEliminated;

  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;

  ids.erase(ids.begin() + idStart, ids.begin() + idLimit);
}

/// Returns the position of the identifier that has the minimum <number of lower
/// bounds> times <number of upper bounds> from the specified range of
/// identifiers [start, end). It is often best to eliminate in the increasing
/// order of these counts when doing Fourier-Motzkin elimination since FM adds
/// that many new constraints.
static unsigned getBestIdToEliminate(const FlatAffineConstraints &cst,
                                     unsigned start, unsigned end) {
  assert(start < cst.getNumIds() && end < cst.getNumIds() + 1);

  auto getProductOfNumLowerUpperBounds = [&](unsigned pos) {
    unsigned numLb = 0;
    unsigned numUb = 0;
    for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
      if (cst.atIneq(r, pos) > 0) {
        ++numLb;
      } else if (cst.atIneq(r, pos) < 0) {
        ++numUb;
      }
    }
    return numLb * numUb;
  };

  unsigned minLoc = start;
  unsigned min = getProductOfNumLowerUpperBounds(start);
  for (unsigned c = start + 1; c < end; c++) {
    unsigned numLbUbProduct = getProductOfNumLowerUpperBounds(c);
    if (numLbUbProduct < min) {
      min = numLbUbProduct;
      minLoc = c;
    }
  }
  return minLoc;
}

// Checks for emptiness of the set by eliminating identifiers successively and
// using the GCD test (on all equality constraints) and checking for trivially
// invalid constraints. Returns 'true' if the constraint system is found to be
// empty; false otherwise.
bool FlatAffineConstraints::isEmpty() const {
  if (isEmptyByGCDTest() || hasInvalidConstraint())
    return true;

  // First, eliminate as many identifiers as possible using Gaussian
  // elimination.
  FlatAffineConstraints tmpCst(*this);
  unsigned currentPos = 0;
  while (currentPos < tmpCst.getNumIds()) {
    tmpCst.gaussianEliminateIds(currentPos, tmpCst.getNumIds());
    ++currentPos;
    // We check emptiness through trivial checks after eliminating each ID to
    // detect emptiness early. Since the checks isEmptyByGCDTest() and
    // hasInvalidConstraint() are linear time and single sweep on the constraint
    // buffer, this appears reasonable - but can optimize in the future.
    if (tmpCst.hasInvalidConstraint() || tmpCst.isEmptyByGCDTest())
      return true;
  }

  // Eliminate the remaining using FM.
  for (unsigned i = 0, e = tmpCst.getNumIds(); i < e; i++) {
    tmpCst.fourierMotzkinEliminate(
        getBestIdToEliminate(tmpCst, 0, tmpCst.getNumIds()));
    // Check for a constraint explosion. This rarely happens in practice, but
    // this check exists as a safeguard against improperly constructed
    // constraint systems or artificially created arbitrarily complex systems
    // that aren't the intended use case for FlatAffineConstraints. This is
    // needed since FM has a worst case exponential complexity in theory.
    if (tmpCst.getNumConstraints() >= kExplosionFactor * getNumIds()) {
      LLVM_DEBUG(llvm::dbgs() << "FM constraint explosion detected\n");
      return false;
    }

    // FM wouldn't have modified the equalities in any way. So no need to again
    // run GCD test. Check for trivial invalid constraints.
    if (tmpCst.hasInvalidConstraint())
      return true;
  }
  return false;
}

// Runs the GCD test on all equality constraints. Returns 'true' if this test
// fails on any equality. Returns 'false' otherwise.
// This test can be used to disprove the existence of a solution. If it returns
// true, no integer solution to the equality constraints can exist.
//
// GCD test definition:
//
// The equality constraint:
//
//  c_1*x_1 + c_2*x_2 + ... + c_n*x_n = c_0
//
// has an integer solution iff:
//
//  GCD of c_1, c_2, ..., c_n divides c_0.
//
bool FlatAffineConstraints::isEmptyByGCDTest() const {
  assert(hasConsistentState());
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    uint64_t gcd = std::abs(atEq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(atEq(i, j)));
    }
    int64_t v = std::abs(atEq(i, numCols - 1));
    if (gcd > 0 && (v % gcd != 0)) {
      return true;
    }
  }
  return false;
}

// Returns a matrix where each row is a vector along which the polytope is
// bounded. The span of the returned vectors is guaranteed to contain all
// such vectors. The returned vectors are NOT guaranteed to be linearly
// independent. This function should not be called on empty sets.
//
// It is sufficient to check the perpendiculars of the constraints, as the set
// of perpendiculars which are bounded must span all bounded directions.
Matrix FlatAffineConstraints::getBoundedDirections() const {
  // Note that it is necessary to add the equalities too (which the constructor
  // does) even though we don't need to check if they are bounded; whether an
  // inequality is bounded or not depends on what other constraints, including
  // equalities, are present.
  Simplex simplex(*this);

  assert(!simplex.isEmpty() && "It is not meaningful to ask whether a "
                               "direction is bounded in an empty set.");

  SmallVector<unsigned, 8> boundedIneqs;
  // The constructor adds the inequalities to the simplex first, so this
  // processes all the inequalities.
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    if (simplex.isBoundedAlongConstraint(i))
      boundedIneqs.push_back(i);
  }

  // The direction vector is given by the coefficients and does not include the
  // constant term, so the matrix has one fewer column.
  unsigned dirsNumCols = getNumCols() - 1;
  Matrix dirs(boundedIneqs.size() + getNumEqualities(), dirsNumCols);

  // Copy the bounded inequalities.
  unsigned row = 0;
  for (unsigned i : boundedIneqs) {
    for (unsigned col = 0; col < dirsNumCols; ++col)
      dirs(row, col) = atIneq(i, col);
    ++row;
  }

  // Copy the equalities. All the equalities' perpendiculars are bounded.
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned col = 0; col < dirsNumCols; ++col)
      dirs(row, col) = atEq(i, col);
    ++row;
  }

  return dirs;
}

bool eqInvolvesSuffixDims(const FlatAffineConstraints &fac, unsigned eqIndex,
                          unsigned numDims) {
  for (unsigned e = fac.getNumIds(), j = e - numDims; j < e; ++j)
    if (fac.atEq(eqIndex, j) != 0)
      return true;
  return false;
}
bool ineqInvolvesSuffixDims(const FlatAffineConstraints &fac,
                            unsigned ineqIndex, unsigned numDims) {
  for (unsigned e = fac.getNumIds(), j = e - numDims; j < e; ++j)
    if (fac.atIneq(ineqIndex, j) != 0)
      return true;
  return false;
}

void removeConstraintsInvolvingSuffixDims(FlatAffineConstraints &fac,
                                          unsigned unboundedDims) {
  // We iterate backwards so that whether we remove constraint i - 1 or not, the
  // next constraint to be tested is always i - 2.
  for (unsigned i = fac.getNumEqualities(); i > 0; i--)
    if (eqInvolvesSuffixDims(fac, i - 1, unboundedDims))
      fac.removeEquality(i - 1);
  for (unsigned i = fac.getNumInequalities(); i > 0; i--)
    if (ineqInvolvesSuffixDims(fac, i - 1, unboundedDims))
      fac.removeInequality(i - 1);
}

bool FlatAffineConstraints::isIntegerEmpty() const {
  return !findIntegerSample().hasValue();
}

/// Let this set be S. If S is bounded then we directly call into the GBR
/// sampling algorithm. Otherwise, there are some unbounded directions, i.e.,
/// vectors v such that S extends to infinity along v or -v. In this case we
/// use an algorithm described in the integer set library (isl) manual and used
/// by the isl_set_sample function in that library. The algorithm is:
///
/// 1) Apply a unimodular transform T to S to obtain S*T, such that all
/// dimensions in which S*T is bounded lie in the linear span of a prefix of the
/// dimensions.
///
/// 2) Construct a set B by removing all constraints that involve
/// the unbounded dimensions and then deleting the unbounded dimensions. Note
/// that B is a Bounded set.
///
/// 3) Try to obtain a sample from B using the GBR sampling
/// algorithm. If no sample is found, return that S is empty.
///
/// 4) Otherwise, substitute the obtained sample into S*T to obtain a set
/// C. C is a full-dimensional Cone and always contains a sample.
///
/// 5) Obtain an integer sample from C.
///
/// 6) Return T*v, where v is the concatenation of the samples from B and C.
///
/// The following is a sketch of a proof that
/// a) If the algorithm returns empty, then S is empty.
/// b) If the algorithm returns a sample, it is a valid sample in S.
///
/// The algorithm returns empty only if B is empty, in which case S*T is
/// certainly empty since B was obtained by removing constraints and then
/// deleting unconstrained dimensions from S*T. Since T is unimodular, a vector
/// v is in S*T iff T*v is in S. So in this case, since
/// S*T is empty, S is empty too.
///
/// Otherwise, the algorithm substitutes the sample from B into S*T. All the
/// constraints of S*T that did not involve unbounded dimensions are satisfied
/// by this substitution. All dimensions in the linear span of the dimensions
/// outside the prefix are unbounded in S*T (step 1). Substituting values for
/// the bounded dimensions cannot make these dimensions bounded, and these are
/// the only remaining dimensions in C, so C is unbounded along every vector (in
/// the positive or negative direction, or both). C is hence a full-dimensional
/// cone and therefore always contains an integer point.
///
/// Concatenating the samples from B and C gives a sample v in S*T, so the
/// returned sample T*v is a sample in S.
Optional<SmallVector<int64_t, 8>>
FlatAffineConstraints::findIntegerSample() const {
  // First, try the GCD test heuristic.
  if (isEmptyByGCDTest())
    return {};

  Simplex simplex(*this);
  if (simplex.isEmpty())
    return {};

  // For a bounded set, we directly call into the GBR sampling algorithm.
  if (!simplex.isUnbounded())
    return simplex.findIntegerSample();

  // The set is unbounded. We cannot directly use the GBR algorithm.
  //
  // m is a matrix containing, in each row, a vector in which S is
  // bounded, such that the linear span of all these dimensions contains all
  // bounded dimensions in S.
  Matrix m = getBoundedDirections();
  // In column echelon form, each row of m occupies only the first rank(m)
  // columns and has zeros on the other columns. The transform T that brings S
  // to column echelon form is unimodular as well, so this is a suitable
  // transform to use in step 1 of the algorithm.
  std::pair<unsigned, LinearTransform> result =
      LinearTransform::makeTransformToColumnEchelon(std::move(m));
  const LinearTransform &transform = result.second;
  // 1) Apply T to S to obtain S*T.
  FlatAffineConstraints transformedSet = transform.applyTo(*this);

  // 2) Remove the unbounded dimensions and constraints involving them to
  // obtain a bounded set.
  FlatAffineConstraints boundedSet = transformedSet;
  unsigned numBoundedDims = result.first;
  unsigned numUnboundedDims = getNumIds() - numBoundedDims;
  removeConstraintsInvolvingSuffixDims(boundedSet, numUnboundedDims);
  boundedSet.removeIdRange(numBoundedDims, boundedSet.getNumIds());

  // 3) Try to obtain a sample from the bounded set.
  Optional<SmallVector<int64_t, 8>> boundedSample =
      Simplex(boundedSet).findIntegerSample();
  if (!boundedSample)
    return {};
  assert(boundedSet.containsPoint(*boundedSample) &&
         "Simplex returned an invalid sample!");

  // 4) Substitute the values of the bounded dimensions into S*T to obtain a
  // full-dimensional cone, which necessarily contains an integer sample.
  transformedSet.setAndEliminate(0, *boundedSample);
  FlatAffineConstraints &cone = transformedSet;

  // 5) Obtain an integer sample from the cone.
  //
  // We shrink the cone such that for any rational point in the shrunken cone,
  // rounding up each of the point's coordinates produces a point that still
  // lies in the original cone.
  //
  // Rounding up a point x adds a number e_i in [0, 1) to each coordinate x_i.
  // For each inequality sum_i a_i x_i + c >= 0 in the original cone, the
  // shrunken cone will have the inequality tightened by some amount s, such
  // that if x satisfies the shrunken cone's tightened inequality, then x + e
  // satisfies the original inequality, i.e.,
  //
  // sum_i a_i x_i + c + s >= 0 implies sum_i a_i (x_i + e_i) + c >= 0
  //
  // for any e_i values in [0, 1). In fact, we will handle the slightly more
  // general case where e_i can be in [0, 1]. For example, consider the
  // inequality 2x_1 - 3x_2 - 7x_3 - 6 >= 0, and let x = (3, 0, 0). How low
  // could the LHS go if we added a number in [0, 1] to each coordinate? The LHS
  // is minimized when we add 1 to the x_i with negative coefficient a_i and
  // keep the other x_i the same. In the example, we would get x = (3, 1, 1),
  // changing the value of the LHS by -3 + -7 = -10.
  //
  // In general, the value of the LHS can change by at most the sum of the
  // negative a_i, so we accomodate this by shifting the inequality by this
  // amount for the shrunken cone.
  for (unsigned i = 0, e = cone.getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0; j < cone.numIds; ++j) {
      int64_t coeff = cone.atIneq(i, j);
      if (coeff < 0)
        cone.atIneq(i, cone.numIds) += coeff;
    }
  }

  // Obtain an integer sample in the cone by rounding up a rational point from
  // the shrunken cone. Shrinking the cone amounts to shifting its apex
  // "inwards" without changing its "shape"; the shrunken cone is still a
  // full-dimensional cone and is hence non-empty.
  Simplex shrunkenConeSimplex(cone);
  assert(!shrunkenConeSimplex.isEmpty() && "Shrunken cone cannot be empty!");
  SmallVector<Fraction, 8> shrunkenConeSample =
      shrunkenConeSimplex.getRationalSample();

  SmallVector<int64_t, 8> coneSample(llvm::map_range(shrunkenConeSample, ceil));

  // 6) Return transform * concat(boundedSample, coneSample).
  SmallVector<int64_t, 8> &sample = boundedSample.getValue();
  sample.append(coneSample.begin(), coneSample.end());
  return transform.preMultiplyColumn(sample);
}

/// Helper to evaluate an affine expression at a point.
/// The expression is a list of coefficients for the dimensions followed by the
/// constant term.
static int64_t valueAt(ArrayRef<int64_t> expr, ArrayRef<int64_t> point) {
  assert(expr.size() == 1 + point.size() &&
         "Dimensionalities of point and expression don't match!");
  int64_t value = expr.back();
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

/// A point satisfies an equality iff the value of the equality at the
/// expression is zero, and it satisfies an inequality iff the value of the
/// inequality at that point is non-negative.
bool FlatAffineConstraints::containsPoint(ArrayRef<int64_t> point) const {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    if (valueAt(getEquality(i), point) != 0)
      return false;
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    if (valueAt(getInequality(i), point) < 0)
      return false;
  }
  return true;
}

/// Tightens inequalities given that we are dealing with integer spaces. This is
/// analogous to the GCD test but applied to inequalities. The constant term can
/// be reduced to the preceding multiple of the GCD of the coefficients, i.e.,
///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
/// fast method - linear in the number of coefficients.
// Example on how this affects practical cases: consider the scenario:
// 64*i >= 100, j = 64*i; without a tightening, elimination of i would yield
// j >= 100 instead of the tighter (exact) j >= 128.
void FlatAffineConstraints::gcdTightenInequalities() {
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    uint64_t gcd = std::abs(atIneq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(atIneq(i, j)));
    }
    if (gcd > 0 && gcd != 1) {
      int64_t gcdI = static_cast<int64_t>(gcd);
      // Tighten the constant term and normalize the constraint by the GCD.
      atIneq(i, numCols - 1) = mlir::floorDiv(atIneq(i, numCols - 1), gcdI);
      for (unsigned j = 0, e = numCols - 1; j < e; ++j)
        atIneq(i, j) /= gcdI;
    }
  }
}

// Eliminates all identifier variables in column range [posStart, posLimit).
// Returns the number of variables eliminated.
unsigned FlatAffineConstraints::gaussianEliminateIds(unsigned posStart,
                                                     unsigned posLimit) {
  // Return if identifier positions to eliminate are out of range.
  assert(posLimit <= numIds);
  assert(hasConsistentState());

  if (posStart >= posLimit)
    return 0;

  gcdTightenInequalities();

  unsigned pivotCol = 0;
  for (pivotCol = posStart; pivotCol < posLimit; ++pivotCol) {
    // Find a row which has a non-zero coefficient in column 'j'.
    unsigned pivotRow;
    if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/true,
                                     &pivotRow)) {
      // No pivot row in equalities with non-zero at 'pivotCol'.
      if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/false,
                                       &pivotRow)) {
        // If inequalities are also non-zero in 'pivotCol', it can be
        // eliminated.
        continue;
      }
      break;
    }

    // Eliminate identifier at 'pivotCol' from each equality row.
    for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/true);
      normalizeConstraintByGCD</*isEq=*/true>(this, i);
    }

    // Eliminate identifier at 'pivotCol' from each inequality row.
    for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/false);
      normalizeConstraintByGCD</*isEq=*/false>(this, i);
    }
    removeEquality(pivotRow);
    gcdTightenInequalities();
  }
  // Update position limit based on number eliminated.
  posLimit = pivotCol;
  // Remove eliminated columns from all constraints.
  removeIdRange(posStart, posLimit);
  return posLimit - posStart;
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
      auto ub = cst.getConstantUpperBound(dimExpr.getPosition());
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

/// Gather all lower and upper bounds of the identifier at `pos`, and
/// optionally any equalities on it. In addition, the bounds are to be
/// independent of identifiers in position range [`offset`, `offset` + `num`).
void FlatAffineConstraints::getLowerAndUpperBoundIndices(
    unsigned pos, SmallVectorImpl<unsigned> *lbIndices,
    SmallVectorImpl<unsigned> *ubIndices, SmallVectorImpl<unsigned> *eqIndices,
    unsigned offset, unsigned num) const {
  assert(pos < getNumIds() && "invalid position");
  assert(offset + num < getNumCols() && "invalid range");

  // Checks for a constraint that has a non-zero coeff for the identifiers in
  // the position range [offset, offset + num) while ignoring `pos`.
  auto containsConstraintDependentOnRange = [&](unsigned r, bool isEq) {
    unsigned c, f;
    auto cst = isEq ? getEquality(r) : getInequality(r);
    for (c = offset, f = offset + num; c < f; ++c) {
      if (c == pos)
        continue;
      if (cst[c] != 0)
        break;
    }
    return c < f;
  };

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    if (containsConstraintDependentOnRange(r, /*isEq=*/false))
      continue;
    if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices->push_back(r);
    } else if (atIneq(r, pos) <= -1) {
      // Upper bound.
      ubIndices->push_back(r);
    }
  }

  // An equality is both a lower and upper bound. Record any equalities
  // involving the pos^th identifier.
  if (!eqIndices)
    return;

  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) == 0)
      continue;
    if (containsConstraintDependentOnRange(r, /*isEq=*/true))
      continue;
    eqIndices->push_back(r);
  }
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

  SmallVector<unsigned, 4> lbIndices, ubIndices;
  cst.getLowerAndUpperBoundIndices(pos, &lbIndices, &ubIndices);

  // Check if any lower bound, upper bound pair is of the form:
  // divisor * id >=  expr - (divisor - 1)    <-- Lower bound for 'id'
  // divisor * id <=  expr                    <-- Upper bound for 'id'
  // Then, 'id' is equivalent to 'expr floordiv divisor'.  (where divisor > 1).
  //
  // For example:
  //    32*k >= 16*i + j - 31                 <-- Lower bound for 'k'
  //    32*k  <= 16*i + j                     <-- Upper bound for 'k'
  //    expr = 16*i + j, divisor = 32
  //    k = ( 16*i + j ) floordiv 32
  //
  //    4q >= i + j - 2                       <-- Lower bound for 'q'
  //    4q <= i + j + 1                       <-- Upper bound for 'q'
  //    expr = i + j + 1, divisor = 4
  //    q = (i + j + 1) floordiv 4
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Due to the form of the inequalities, the sum of constants of upper
      // bound and lower bound is divisor - 1. The 'divisor' here is
      // cst.atIneq(lbPos, pos) and we already know that it's positive (since
      // cst.Ineq(lbPos, ...) is a lower bound expr for 'pos'.
      // Check if this sum of constants is divisor - 1.
      int64_t divisor = cst.atIneq(lbPos, pos);
      int64_t constantSum = cst.atIneq(lbPos, cst.getNumCols() - 1) +
                            cst.atIneq(ubPos, cst.getNumCols() - 1);
      if (constantSum != divisor - 1)
        continue;
      // For the remaining part, check if the lower bound expr's coeff's are
      // negations of corresponding upper bound ones'.
      unsigned c, f;
      for (c = 0, f = cst.getNumCols() - 1; c < f; ++c)
        if (cst.atIneq(lbPos, c) != -cst.atIneq(ubPos, c))
          break;
      // Lb coeff's aren't negative of ub coeff's (for the non constant term
      // part).
      if (c < f)
        continue;
      // Due to the form of the upper bound inequality, the constant term of
      // `expr` is the constant term of upper bound inequality.
      int64_t divConstantTerm = cst.atIneq(ubPos, cst.getNumCols() - 1);
      // Construct the dividend expression.
      auto dividendExpr = getAffineConstantExpr(divConstantTerm, context);
      for (c = 0, f = cst.getNumCols() - 1; c < f; ++c) {
        if (c == pos)
          continue;
        int64_t ubVal = cst.atIneq(ubPos, c);
        if (ubVal == 0)
          continue;
        if (!exprs[c])
          break;
        dividendExpr = dividendExpr + ubVal * exprs[c];
      }
      // Expression can't be constructed as it depends on a yet unknown
      // identifier.
      // TODO: Visit/compute the identifiers in an order so that this doesn't
      // happen. More complex but much more efficient.
      if (c < f)
        continue;
      // Successfully detected the floordiv.
      exprs[pos] = dividendExpr.floorDiv(divisor);
      return true;
    }
  }
  return false;
}

// Fills an inequality row with the value 'val'.
static inline void fillInequality(FlatAffineConstraints *cst, unsigned r,
                                  int64_t val) {
  for (unsigned c = 0, f = cst->getNumCols(); c < f; c++) {
    cst->atIneq(r, c) = val;
  }
}

// Negates an inequality.
static inline void negateInequality(FlatAffineConstraints *cst, unsigned r) {
  for (unsigned c = 0, f = cst->getNumCols(); c < f; c++) {
    cst->atIneq(r, c) = -cst->atIneq(r, c);
  }
}

// A more complex check to eliminate redundant inequalities. Uses FourierMotzkin
// to check if a constraint is redundant.
void FlatAffineConstraints::removeRedundantInequalities() {
  SmallVector<bool, 32> redun(getNumInequalities(), false);
  // To check if an inequality is redundant, we replace the inequality by its
  // complement (for eg., i - 1 >= 0 by i <= 0), and check if the resulting
  // system is empty. If it is, the inequality is redundant.
  FlatAffineConstraints tmpCst(*this);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // Change the inequality to its complement.
    negateInequality(&tmpCst, r);
    tmpCst.atIneq(r, tmpCst.getNumCols() - 1)--;
    if (tmpCst.isEmpty()) {
      redun[r] = true;
      // Zero fill the redundant inequality.
      fillInequality(this, r, /*val=*/0);
      fillInequality(&tmpCst, r, /*val=*/0);
    } else {
      // Reverse the change (to avoid recreating tmpCst each time).
      tmpCst.atIneq(r, tmpCst.getNumCols() - 1)++;
      negateInequality(&tmpCst, r);
    }
  }

  // Scan to get rid of all rows marked redundant, in-place.
  auto copyRow = [&](unsigned src, unsigned dest) {
    if (src == dest)
      return;
    for (unsigned c = 0, e = getNumCols(); c < e; c++) {
      atIneq(dest, c) = atIneq(src, c);
    }
  };
  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (!redun[r])
      copyRow(r, pos++);
  }
  inequalities.resizeVertically(pos);
}

// A more complex check to eliminate redundant inequalities and equalities. Uses
// Simplex to check if a constraint is redundant.
void FlatAffineConstraints::removeRedundantConstraints() {
  // First, we run gcdTightenInequalities. This allows us to catch some
  // constraints which are not redundant when considering rational solutions
  // but are redundant in terms of integer solutions.
  gcdTightenInequalities();
  Simplex simplex(*this);
  simplex.detectRedundant();

  auto copyInequality = [&](unsigned src, unsigned dest) {
    if (src == dest)
      return;
    for (unsigned c = 0, e = getNumCols(); c < e; c++)
      atIneq(dest, c) = atIneq(src, c);
  };
  unsigned pos = 0;
  unsigned numIneqs = getNumInequalities();
  // Scan to get rid of all inequalities marked redundant, in-place. In Simplex,
  // the first constraints added are the inequalities.
  for (unsigned r = 0; r < numIneqs; r++) {
    if (!simplex.isMarkedRedundant(r))
      copyInequality(r, pos++);
  }
  inequalities.resizeVertically(pos);

  // Scan to get rid of all equalities marked redundant, in-place. In Simplex,
  // after the inequalities, a pair of constraints for each equality is added.
  // An equality is redundant if both the inequalities in its pair are
  // redundant.
  auto copyEquality = [&](unsigned src, unsigned dest) {
    if (src == dest)
      return;
    for (unsigned c = 0, e = getNumCols(); c < e; c++)
      atEq(dest, c) = atEq(src, c);
  };
  pos = 0;
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (!(simplex.isMarkedRedundant(numIneqs + 2 * r) &&
          simplex.isMarkedRedundant(numIneqs + 2 * r + 1)))
      copyEquality(r, pos++);
  }
  equalities.resizeVertically(pos);
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

      auto lbConst = getConstantLowerBound(pos);
      auto ubConst = getConstantUpperBound(pos);
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
      if (!findConstraintWithNonZeroAt(*this, pos, /*isEq=*/true, &idx)) {
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
        auto lbConst = getConstantLowerBound(pos + offset);
        if (lbConst.hasValue()) {
          lbMap = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(lbConst.getValue(), context));
        }
      }
      if (!ubMap || ubMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice ub\n");
        auto ubConst = getConstantUpperBound(pos + offset);
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

LogicalResult
FlatAffineConstraints::addLowerOrUpperBound(unsigned pos, AffineMap boundMap,
                                            ValueRange boundOperands, bool eq,
                                            bool lower) {
  assert(pos < getNumDimAndSymbolIds() && "invalid position");
  // Equality follows the logic of lower bound except that we add an equality
  // instead of an inequality.
  assert((!eq || boundMap.getNumResults() == 1) && "single result expected");
  if (eq)
    lower = true;

  // Fully compose map and operands; canonicalize and simplify so that we
  // transitively get to terminal symbols or loop IVs.
  auto map = boundMap;
  SmallVector<Value, 4> operands(boundOperands.begin(), boundOperands.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  for (auto operand : operands)
    addInductionVarOrTerminalSymbol(operand);

  FlatAffineConstraints localVarCst;
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localVarCst))) {
    LLVM_DEBUG(llvm::dbgs() << "semi-affine expressions not yet supported\n");
    return failure();
  }

  // Merge and align with localVarCst.
  if (localVarCst.getNumLocalIds() > 0) {
    // Set values for localVarCst.
    localVarCst.setIdValues(0, localVarCst.getNumDimAndSymbolIds(), operands);
    for (auto operand : operands) {
      unsigned pos;
      if (findId(operand, &pos)) {
        if (pos >= getNumDimIds() && pos < getNumDimAndSymbolIds()) {
          // If the local var cst has this as a dim, turn it into its symbol.
          turnDimIntoSymbol(&localVarCst, operand);
        } else if (pos < getNumDimIds()) {
          // Or vice versa.
          turnSymbolIntoDim(&localVarCst, operand);
        }
      }
    }
    mergeAndAlignIds(/*offset=*/0, this, &localVarCst);
    append(localVarCst);
  }

  // Record positions of the operands in the constraint system. Need to do
  // this here since the constraint system changes after a bound is added.
  SmallVector<unsigned, 8> positions;
  unsigned numOperands = operands.size();
  for (auto operand : operands) {
    unsigned pos;
    if (!findId(operand, &pos))
      assert(0 && "expected to be found");
    positions.push_back(pos);
  }

  for (const auto &flatExpr : flatExprs) {
    SmallVector<int64_t, 4> ineq(getNumCols(), 0);
    ineq[pos] = lower ? 1 : -1;
    // Dims and symbols.
    for (unsigned j = 0, e = map.getNumInputs(); j < e; j++) {
      ineq[positions[j]] = lower ? -flatExpr[j] : flatExpr[j];
    }
    // Copy over the local id coefficients.
    unsigned numLocalIds = flatExpr.size() - 1 - numOperands;
    for (unsigned jj = 0, j = getNumIds() - numLocalIds; jj < numLocalIds;
         jj++, j++) {
      ineq[j] =
          lower ? -flatExpr[numOperands + jj] : flatExpr[numOperands + jj];
    }
    // Constant term.
    ineq[getNumCols() - 1] =
        lower ? -flatExpr[flatExpr.size() - 1]
              // Upper bound in flattenedExpr is an exclusive one.
              : flatExpr[flatExpr.size() - 1] - 1;
    eq ? addEquality(ineq) : addInequality(ineq);
  }
  return success();
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
LogicalResult FlatAffineConstraints::addSliceBounds(ArrayRef<Value> values,
                                                    ArrayRef<AffineMap> lbMaps,
                                                    ArrayRef<AffineMap> ubMaps,
                                                    ArrayRef<Value> operands) {
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
      if (failed(addLowerOrUpperBound(pos, lbMap, operands, /*eq=*/true,
                                      /*lower=*/true)))
        return failure();
      continue;
    }

    // If lower or upper bound maps are null or provide no results, it implies
    // that the source loop was not at all sliced, and the entire loop will be a
    // part of the slice.
    if (lbMap && lbMap.getNumResults() != 0 && ubMap &&
        ubMap.getNumResults() != 0) {
      if (failed(addLowerOrUpperBound(pos, lbMap, operands, /*eq=*/false,
                                      /*lower=*/true)))
        return failure();
      if (failed(addLowerOrUpperBound(pos, ubMap, operands, /*eq=*/false,
                                      /*lower=*/false)))
        return failure();
    } else {
      auto loop = getForInductionVarOwner(values[i]);
      if (failed(this->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

void FlatAffineConstraints::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned row = equalities.appendExtraRow();
  for (unsigned i = 0, e = eq.size(); i < e; ++i)
    equalities(row, i) = eq[i];
}

void FlatAffineConstraints::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = inEq.size(); i < e; ++i)
    inequalities(row, i) = inEq[i];
}

void FlatAffineConstraints::addConstantLowerBound(unsigned pos, int64_t lb) {
  assert(pos < getNumCols());
  unsigned row = inequalities.appendExtraRow();
  inequalities(row, pos) = 1;
  inequalities(row, getNumCols() - 1) = -lb;
}

void FlatAffineConstraints::addConstantUpperBound(unsigned pos, int64_t ub) {
  assert(pos < getNumCols());
  unsigned row = inequalities.appendExtraRow();
  inequalities(row, pos) = -1;
  inequalities(row, getNumCols() - 1) = ub;
}

void FlatAffineConstraints::addConstantLowerBound(ArrayRef<int64_t> expr,
                                                  int64_t lb) {
  addInequality(expr);
  inequalities(inequalities.getNumRows() - 1, getNumCols() - 1) += -lb;
}

void FlatAffineConstraints::addConstantUpperBound(ArrayRef<int64_t> expr,
                                                  int64_t ub) {
  assert(expr.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = expr.size(); i < e; ++i)
    inequalities(row, i) = -expr[i];
  inequalities(inequalities.getNumRows() - 1, getNumCols() - 1) += ub;
}

/// Adds a new local identifier as the floordiv of an affine function of other
/// identifiers, the coefficients of which are provided in 'dividend' and with
/// respect to a positive constant 'divisor'. Two constraints are added to the
/// system to capture equivalence with the floordiv.
///      q = expr floordiv c    <=>   c*q <= expr <= c*q + c - 1.
void FlatAffineConstraints::addLocalFloorDiv(ArrayRef<int64_t> dividend,
                                             int64_t divisor) {
  assert(dividend.size() == getNumCols() && "incorrect dividend size");
  assert(divisor > 0 && "positive divisor expected");

  addLocalId(getNumLocalIds());

  // Add two constraints for this new identifier 'q'.
  SmallVector<int64_t, 8> bound(dividend.size() + 1);

  // dividend - q * divisor >= 0
  std::copy(dividend.begin(), dividend.begin() + dividend.size() - 1,
            bound.begin());
  bound.back() = dividend.back();
  bound[getNumIds() - 1] = -divisor;
  addInequality(bound);

  // -dividend +qdivisor * q + divisor - 1 >= 0
  std::transform(bound.begin(), bound.end(), bound.begin(),
                 std::negate<int64_t>());
  bound[bound.size() - 1] += divisor - 1;
  addInequality(bound);
}

bool FlatAffineConstraints::findId(Value id, unsigned *pos) const {
  unsigned i = 0;
  for (const auto &mayBeId : ids) {
    if (mayBeId.hasValue() && mayBeId.getValue() == id) {
      *pos = i;
      return true;
    }
    i++;
  }
  return false;
}

bool FlatAffineConstraints::containsId(Value id) const {
  return llvm::any_of(ids, [&](const Optional<Value> &mayBeId) {
    return mayBeId.hasValue() && mayBeId.getValue() == id;
  });
}

void FlatAffineConstraints::swapId(unsigned posA, unsigned posB) {
  assert(posA < getNumIds() && "invalid position A");
  assert(posB < getNumIds() && "invalid position B");

  if (posA == posB)
    return;

  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    std::swap(atIneq(r, posA), atIneq(r, posB));
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++)
    std::swap(atEq(r, posA), atEq(r, posB));
  std::swap(getId(posA), getId(posB));
}

void FlatAffineConstraints::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(newSymbolCount <= numDims + numSymbols &&
         "invalid separation position");
  numDims = numDims + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}

/// Sets the specified identifier to a constant value.
void FlatAffineConstraints::setIdToConstant(unsigned pos, int64_t val) {
  equalities.resizeVertically(equalities.getNumRows() + 1);
  unsigned row = equalities.getNumRows() - 1;
  equalities(row, pos) = 1;
  equalities(row, getNumCols() - 1) = -val;
}

/// Sets the specified identifier to a constant value; asserts if the id is not
/// found.
void FlatAffineConstraints::setIdToConstant(Value id, int64_t val) {
  unsigned pos;
  if (!findId(id, &pos))
    // This is a pre-condition for this method.
    assert(0 && "id not found");
  setIdToConstant(pos, val);
}

void FlatAffineConstraints::removeEquality(unsigned pos) {
  equalities.removeRow(pos);
}

void FlatAffineConstraints::removeInequality(unsigned pos) {
  inequalities.removeRow(pos);
}

/// Finds an equality that equates the specified identifier to a constant.
/// Returns the position of the equality row. If 'symbolic' is set to true,
/// symbols are also treated like a constant, i.e., an affine function of the
/// symbols is also treated like a constant. Returns -1 if such an equality
/// could not be found.
static int findEqualityToConstant(const FlatAffineConstraints &cst,
                                  unsigned pos, bool symbolic = false) {
  assert(pos < cst.getNumIds() && "invalid position");
  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    int64_t v = cst.atEq(r, pos);
    if (v * v != 1)
      continue;
    unsigned c;
    unsigned f = symbolic ? cst.getNumDimIds() : cst.getNumIds();
    // This checks for zeros in all positions other than 'pos' in [0, f)
    for (c = 0; c < f; c++) {
      if (c == pos)
        continue;
      if (cst.atEq(r, c) != 0) {
        // Dependent on another identifier.
        break;
      }
    }
    if (c == f)
      // Equality is free of other identifiers.
      return r;
  }
  return -1;
}

void FlatAffineConstraints::setAndEliminate(unsigned pos,
                                            ArrayRef<int64_t> values) {
  if (values.empty())
    return;
  assert(pos + values.size() <= getNumIds() &&
         "invalid position or too many values");
  // Setting x_j = p in sum_i a_i x_i + c is equivalent to adding p*a_j to the
  // constant term and removing the id x_j. We do this for all the ids
  // pos, pos + 1, ... pos + values.size() - 1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
      atIneq(r, getNumCols() - 1) += atIneq(r, pos + i) * values[i];
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++)
    for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
      atEq(r, getNumCols() - 1) += atEq(r, pos + i) * values[i];
  removeIdRange(pos, pos + values.size());
}

LogicalResult FlatAffineConstraints::constantFoldId(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  int rowIdx;
  if ((rowIdx = findEqualityToConstant(*this, pos)) == -1)
    return failure();

  // atEq(rowIdx, pos) is either -1 or 1.
  assert(atEq(rowIdx, pos) * atEq(rowIdx, pos) == 1);
  int64_t constVal = -atEq(rowIdx, getNumCols() - 1) / atEq(rowIdx, pos);
  setAndEliminate(pos, constVal);
  return success();
}

void FlatAffineConstraints::constantFoldIdRange(unsigned pos, unsigned num) {
  for (unsigned s = pos, t = pos, e = pos + num; s < e; s++) {
    if (failed(constantFoldId(t)))
      t++;
  }
}

/// Returns a non-negative constant bound on the extent (upper bound - lower
/// bound) of the specified identifier if it is found to be a constant; returns
/// None if it's not a constant. This methods treats symbolic identifiers
/// specially, i.e., it looks for constant differences between affine
/// expressions involving only the symbolic identifiers. See comments at
/// function definition for example. 'lb', if provided, is set to the lower
/// bound associated with the constant difference. Note that 'lb' is purely
/// symbolic and thus will contain the coefficients of the symbolic identifiers
/// and the constant coefficient.
//  Egs: 0 <= i <= 15, return 16.
//       s0 + 2 <= i <= s0 + 17, returns 16. (s0 has to be a symbol)
//       s0 + s1 + 16 <= d0 <= s0 + s1 + 31, returns 16.
//       s0 - 7 <= 8*j <= s0 returns 1 with lb = s0, lbDivisor = 8 (since lb =
//       ceil(s0 - 7 / 8) = floor(s0 / 8)).
Optional<int64_t> FlatAffineConstraints::getConstantBoundOnDimSize(
    unsigned pos, SmallVectorImpl<int64_t> *lb, int64_t *boundFloorDivisor,
    SmallVectorImpl<int64_t> *ub, unsigned *minLbPos,
    unsigned *minUbPos) const {
  assert(pos < getNumDimIds() && "Invalid identifier position");

  // Find an equality for 'pos'^th identifier that equates it to some function
  // of the symbolic identifiers (+ constant).
  int eqPos = findEqualityToConstant(*this, pos, /*symbolic=*/true);
  if (eqPos != -1) {
    auto eq = getEquality(eqPos);
    // If the equality involves a local var, punt for now.
    // TODO: this can be handled in the future by using the explicit
    // representation of the local vars.
    if (!std::all_of(eq.begin() + getNumDimAndSymbolIds(), eq.end() - 1,
                     [](int64_t coeff) { return coeff == 0; }))
      return None;

    // This identifier can only take a single value.
    if (lb) {
      // Set lb to that symbolic value.
      lb->resize(getNumSymbolIds() + 1);
      if (ub)
        ub->resize(getNumSymbolIds() + 1);
      for (unsigned c = 0, f = getNumSymbolIds() + 1; c < f; c++) {
        int64_t v = atEq(eqPos, pos);
        // atEq(eqRow, pos) is either -1 or 1.
        assert(v * v == 1);
        (*lb)[c] = v < 0 ? atEq(eqPos, getNumDimIds() + c) / -v
                         : -atEq(eqPos, getNumDimIds() + c) / v;
        // Since this is an equality, ub = lb.
        if (ub)
          (*ub)[c] = (*lb)[c];
      }
      assert(boundFloorDivisor &&
             "both lb and divisor or none should be provided");
      *boundFloorDivisor = 1;
    }
    if (minLbPos)
      *minLbPos = eqPos;
    if (minUbPos)
      *minUbPos = eqPos;
    return 1;
  }

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  // Positions of constraints that are lower/upper bounds on the variable.
  SmallVector<unsigned, 4> lbIndices, ubIndices;

  // Gather all symbolic lower bounds and upper bounds of the variable, i.e.,
  // the bounds can only involve symbolic (and local) identifiers. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  getLowerAndUpperBoundIndices(pos, &lbIndices, &ubIndices,
                               /*eqIndices=*/nullptr, /*offset=*/0,
                               /*num=*/getNumDimIds());

  Optional<int64_t> minDiff = None;
  unsigned minLbPosition = 0, minUbPosition = 0;
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Look for a lower bound and an upper bound that only differ by a
      // constant, i.e., pairs of the form  0 <= c_pos - f(c_i's) <= diffConst.
      // For example, if ii is the pos^th variable, we are looking for
      // constraints like ii >= i, ii <= ii + 50, 50 being the difference. The
      // minimum among all such constant differences is kept since that's the
      // constant bounding the extent of the pos^th variable.
      unsigned j, e;
      for (j = 0, e = getNumCols() - 1; j < e; j++)
        if (atIneq(ubPos, j) != -atIneq(lbPos, j)) {
          break;
        }
      if (j < getNumCols() - 1)
        continue;
      int64_t diff = ceilDiv(atIneq(ubPos, getNumCols() - 1) +
                                 atIneq(lbPos, getNumCols() - 1) + 1,
                             atIneq(lbPos, pos));
      // This bound is non-negative by definition.
      diff = std::max<int64_t>(diff, 0);
      if (minDiff == None || diff < minDiff) {
        minDiff = diff;
        minLbPosition = lbPos;
        minUbPosition = ubPos;
      }
    }
  }
  if (lb && minDiff.hasValue()) {
    // Set lb to the symbolic lower bound.
    lb->resize(getNumSymbolIds() + 1);
    if (ub)
      ub->resize(getNumSymbolIds() + 1);
    // The lower bound is the ceildiv of the lb constraint over the coefficient
    // of the variable at 'pos'. We express the ceildiv equivalently as a floor
    // for uniformity. For eg., if the lower bound constraint was: 32*d0 - N +
    // 31 >= 0, the lower bound for d0 is ceil(N - 31, 32), i.e., floor(N, 32).
    *boundFloorDivisor = atIneq(minLbPosition, pos);
    assert(*boundFloorDivisor == -atIneq(minUbPosition, pos));
    for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++) {
      (*lb)[c] = -atIneq(minLbPosition, getNumDimIds() + c);
    }
    if (ub) {
      for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++)
        (*ub)[c] = atIneq(minUbPosition, getNumDimIds() + c);
    }
    // The lower bound leads to a ceildiv while the upper bound is a floordiv
    // whenever the coefficient at pos != 1. ceildiv (val / d) = floordiv (val +
    // d - 1 / d); hence, the addition of 'atIneq(minLbPosition, pos) - 1' to
    // the constant term for the lower bound.
    (*lb)[getNumSymbolIds()] += atIneq(minLbPosition, pos) - 1;
  }
  if (minLbPos)
    *minLbPos = minLbPosition;
  if (minUbPos)
    *minUbPos = minUbPosition;
  return minDiff;
}

template <bool isLower>
Optional<int64_t>
FlatAffineConstraints::computeConstantLowerOrUpperBound(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  // Project to 'pos'.
  projectOut(0, pos);
  projectOut(1, getNumIds() - 1);
  // Check if there's an equality equating the '0'^th identifier to a constant.
  int eqRowIdx = findEqualityToConstant(*this, 0, /*symbolic=*/false);
  if (eqRowIdx != -1)
    // atEq(rowIdx, 0) is either -1 or 1.
    return -atEq(eqRowIdx, getNumCols() - 1) / atEq(eqRowIdx, 0);

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, 0) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  Optional<int64_t> minOrMaxConst = None;

  // Take the max across all const lower bounds (or min across all constant
  // upper bounds).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (isLower) {
      if (atIneq(r, 0) <= 0)
        // Not a lower bound.
        continue;
    } else if (atIneq(r, 0) >= 0) {
      // Not an upper bound.
      continue;
    }
    unsigned c, f;
    for (c = 0, f = getNumCols() - 1; c < f; c++)
      if (c != 0 && atIneq(r, c) != 0)
        break;
    if (c < getNumCols() - 1)
      // Not a constant bound.
      continue;

    int64_t boundConst =
        isLower ? mlir::ceilDiv(-atIneq(r, getNumCols() - 1), atIneq(r, 0))
                : mlir::floorDiv(atIneq(r, getNumCols() - 1), -atIneq(r, 0));
    if (isLower) {
      if (minOrMaxConst == None || boundConst > minOrMaxConst)
        minOrMaxConst = boundConst;
    } else {
      if (minOrMaxConst == None || boundConst < minOrMaxConst)
        minOrMaxConst = boundConst;
    }
  }
  return minOrMaxConst;
}

Optional<int64_t>
FlatAffineConstraints::getConstantLowerBound(unsigned pos) const {
  FlatAffineConstraints tmpCst(*this);
  return tmpCst.computeConstantLowerOrUpperBound</*isLower=*/true>(pos);
}

Optional<int64_t>
FlatAffineConstraints::getConstantUpperBound(unsigned pos) const {
  FlatAffineConstraints tmpCst(*this);
  return tmpCst.computeConstantLowerOrUpperBound</*isLower=*/false>(pos);
}

// A simple (naive and conservative) check for hyper-rectangularity.
bool FlatAffineConstraints::isHyperRectangular(unsigned pos,
                                               unsigned num) const {
  assert(pos < getNumCols() - 1);
  // Check for two non-zero coefficients in the range [pos, pos + sum).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atIneq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atEq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  return true;
}

void FlatAffineConstraints::print(raw_ostream &os) const {
  assert(hasConsistentState());
  os << "\nConstraints (" << getNumDimIds() << " dims, " << getNumSymbolIds()
     << " symbols, " << getNumLocalIds() << " locals), (" << getNumConstraints()
     << " constraints)\n";
  os << "(";
  for (unsigned i = 0, e = getNumIds(); i < e; i++) {
    if (ids[i] == None)
      os << "None ";
    else
      os << "Value ";
  }
  os << " const)\n";
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atEq(i, j) << " ";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atIneq(i, j) << " ";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void FlatAffineConstraints::dump() const { print(llvm::errs()); }

/// Removes duplicate constraints, trivially true constraints, and constraints
/// that can be detected as redundant as a result of differing only in their
/// constant term part. A constraint of the form <non-negative constant> >= 0 is
/// considered trivially true.
//  Uses a DenseSet to hash and detect duplicates followed by a linear scan to
//  remove duplicates in place.
void FlatAffineConstraints::removeTrivialRedundancy() {
  gcdTightenInequalities();
  normalizeConstraintsByGCD();

  // A map used to detect redundancy stemming from constraints that only differ
  // in their constant term. The value stored is <row position, const term>
  // for a given row.
  SmallDenseMap<ArrayRef<int64_t>, std::pair<unsigned, int64_t>>
      rowsWithoutConstTerm;
  // To unique rows.
  SmallDenseSet<ArrayRef<int64_t>, 8> rowSet;

  // Check if constraint is of the form <non-negative-constant> >= 0.
  auto isTriviallyValid = [&](unsigned r) -> bool {
    for (unsigned c = 0, e = getNumCols() - 1; c < e; c++) {
      if (atIneq(r, c) != 0)
        return false;
    }
    return atIneq(r, getNumCols() - 1) >= 0;
  };

  // Detect and mark redundant constraints.
  SmallVector<bool, 256> redunIneq(getNumInequalities(), false);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    int64_t *rowStart = &inequalities(r, 0);
    auto row = ArrayRef<int64_t>(rowStart, getNumCols());
    if (isTriviallyValid(r) || !rowSet.insert(row).second) {
      redunIneq[r] = true;
      continue;
    }

    // Among constraints that only differ in the constant term part, mark
    // everything other than the one with the smallest constant term redundant.
    // (eg: among i - 16j - 5 >= 0, i - 16j - 1 >=0, i - 16j - 7 >= 0, the
    // former two are redundant).
    int64_t constTerm = atIneq(r, getNumCols() - 1);
    auto rowWithoutConstTerm = ArrayRef<int64_t>(rowStart, getNumCols() - 1);
    const auto &ret =
        rowsWithoutConstTerm.insert({rowWithoutConstTerm, {r, constTerm}});
    if (!ret.second) {
      // Check if the other constraint has a higher constant term.
      auto &val = ret.first->second;
      if (val.second > constTerm) {
        // The stored row is redundant. Mark it so, and update with this one.
        redunIneq[val.first] = true;
        val = {r, constTerm};
      } else {
        // The one stored makes this one redundant.
        redunIneq[r] = true;
      }
    }
  }

  // Scan to get rid of all rows marked redundant, in-place.
  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    if (!redunIneq[r])
      inequalities.copyRow(r, pos++);

  inequalities.resizeVertically(pos);

  // TODO: consider doing this for equalities as well, but probably not worth
  // the savings.
}

void FlatAffineConstraints::clearAndCopyFrom(
    const FlatAffineConstraints &other) {
  FlatAffineConstraints copy(other);
  std::swap(*this, copy);
  assert(copy.getNumIds() == copy.getIds().size());
}

void FlatAffineConstraints::removeId(unsigned pos) {
  removeIdRange(pos, pos + 1);
}

static std::pair<unsigned, unsigned>
getNewNumDimsSymbols(unsigned pos, const FlatAffineConstraints &cst) {
  unsigned numDims = cst.getNumDimIds();
  unsigned numSymbols = cst.getNumSymbolIds();
  unsigned newNumDims, newNumSymbols;
  if (pos < numDims) {
    newNumDims = numDims - 1;
    newNumSymbols = numSymbols;
  } else if (pos < numDims + numSymbols) {
    assert(numSymbols >= 1);
    newNumDims = numDims;
    newNumSymbols = numSymbols - 1;
  } else {
    newNumDims = numDims;
    newNumSymbols = numSymbols;
  }
  return {newNumDims, newNumSymbols};
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "fm"

/// Eliminates identifier at the specified position using Fourier-Motzkin
/// variable elimination. This technique is exact for rational spaces but
/// conservative (in "rare" cases) for integer spaces. The operation corresponds
/// to a projection operation yielding the (convex) set of integer points
/// contained in the rational shadow of the set. An emptiness test that relies
/// on this method will guarantee emptiness, i.e., it disproves the existence of
/// a solution if it says it's empty.
/// If a non-null isResultIntegerExact is passed, it is set to true if the
/// result is also integer exact. If it's set to false, the obtained solution
/// *may* not be exact, i.e., it may contain integer points that do not have an
/// integer pre-image in the original set.
///
/// Eg:
/// j >= 0, j <= i + 1
/// i >= 0, i <= N + 1
/// Eliminating i yields,
///   j >= 0, 0 <= N + 1, j - 1 <= N + 1
///
/// If darkShadow = true, this method computes the dark shadow on elimination;
/// the dark shadow is a convex integer subset of the exact integer shadow. A
/// non-empty dark shadow proves the existence of an integer solution. The
/// elimination in such a case could however be an under-approximation, and thus
/// should not be used for scanning sets or used by itself for dependence
/// checking.
///
/// Eg: 2-d set, * represents grid points, 'o' represents a point in the set.
///            ^
///            |
///            | * * * * o o
///         i  | * * o o o o
///            | o * * * * *
///            --------------->
///                 j ->
///
/// Eliminating i from this system (projecting on the j dimension):
/// rational shadow / integer light shadow:  1 <= j <= 6
/// dark shadow:                             3 <= j <= 6
/// exact integer shadow:                    j = 1 \union  3 <= j <= 6
/// holes/splinters:                         j = 2
///
/// darkShadow = false, isResultIntegerExact = nullptr are default values.
// TODO: a slight modification to yield dark shadow version of FM (tightened),
// which can prove the existence of a solution if there is one.
void FlatAffineConstraints::fourierMotzkinEliminate(
    unsigned pos, bool darkShadow, bool *isResultIntegerExact) {
  LLVM_DEBUG(llvm::dbgs() << "FM input (eliminate pos " << pos << "):\n");
  LLVM_DEBUG(dump());
  assert(pos < getNumIds() && "invalid position");
  assert(hasConsistentState());

  // Check if this identifier can be eliminated through a substitution.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      LogicalResult ret = gaussianEliminateId(pos);
      (void)ret;
      assert(succeeded(ret) && "Gaussian elimination guaranteed to succeed");
      LLVM_DEBUG(llvm::dbgs() << "FM output (through Gaussian elimination):\n");
      LLVM_DEBUG(dump());
      return;
    }
  }

  // A fast linear time tightening.
  gcdTightenInequalities();

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == getNumInequalities()) {
    // If it doesn't appear, just remove the column and return.
    // TODO: refactor removeColumns to use it from here.
    removeId(pos);
    LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
    LLVM_DEBUG(dump());
    return;
  }

  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> lbIndices;
  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> ubIndices;
  // Positions of constraints that do not involve the variable.
  std::vector<unsigned> nbIndices;
  nbIndices.reserve(getNumInequalities());

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) == 0) {
      // Id does not appear in bound.
      nbIndices.push_back(r);
    } else if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices.push_back(r);
    } else {
      // Upper bound.
      ubIndices.push_back(r);
    }
  }

  // Set the number of dimensions, symbols in the resulting system.
  const auto &dimsSymbols = getNewNumDimsSymbols(pos, *this);
  unsigned newNumDims = dimsSymbols.first;
  unsigned newNumSymbols = dimsSymbols.second;

  SmallVector<Optional<Value>, 8> newIds;
  newIds.reserve(numIds - 1);
  newIds.append(ids.begin(), ids.begin() + pos);
  newIds.append(ids.begin() + pos + 1, ids.end());

  /// Create the new system which has one identifier less.
  FlatAffineConstraints newFac(
      lbIndices.size() * ubIndices.size() + nbIndices.size(),
      getNumEqualities(), getNumCols() - 1, newNumDims, newNumSymbols,
      /*numLocals=*/getNumIds() - 1 - newNumDims - newNumSymbols, newIds);

  assert(newFac.getIds().size() == newFac.getNumIds());

  // This will be used to check if the elimination was integer exact.
  unsigned lcmProducts = 1;

  // Let x be the variable we are eliminating.
  // For each lower bound, lb <= c_l*x, and each upper bound c_u*x <= ub, (note
  // that c_l, c_u >= 1) we have:
  // lb*lcm(c_l, c_u)/c_l <= lcm(c_l, c_u)*x <= ub*lcm(c_l, c_u)/c_u
  // We thus generate a constraint:
  // lcm(c_l, c_u)/c_l*lb <= lcm(c_l, c_u)/c_u*ub.
  // Note if c_l = c_u = 1, all integer points captured by the resulting
  // constraint correspond to integer points in the original system (i.e., they
  // have integer pre-images). Hence, if the lcm's are all 1, the elimination is
  // integer exact.
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      SmallVector<int64_t, 4> ineq;
      ineq.reserve(newFac.getNumCols());
      int64_t lbCoeff = atIneq(lbPos, pos);
      // Note that in the comments above, ubCoeff is the negation of the
      // coefficient in the canonical form as the view taken here is that of the
      // term being moved to the other size of '>='.
      int64_t ubCoeff = -atIneq(ubPos, pos);
      // TODO: refactor this loop to avoid all branches inside.
      for (unsigned l = 0, e = getNumCols(); l < e; l++) {
        if (l == pos)
          continue;
        assert(lbCoeff >= 1 && ubCoeff >= 1 && "bounds wrongly identified");
        int64_t lcm = mlir::lcm(lbCoeff, ubCoeff);
        ineq.push_back(atIneq(ubPos, l) * (lcm / ubCoeff) +
                       atIneq(lbPos, l) * (lcm / lbCoeff));
        lcmProducts *= lcm;
      }
      if (darkShadow) {
        // The dark shadow is a convex subset of the exact integer shadow. If
        // there is a point here, it proves the existence of a solution.
        ineq[ineq.size() - 1] += lbCoeff * ubCoeff - lbCoeff - ubCoeff + 1;
      }
      // TODO: we need to have a way to add inequalities in-place in
      // FlatAffineConstraints instead of creating and copying over.
      newFac.addInequality(ineq);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FM isResultIntegerExact: " << (lcmProducts == 1)
                          << "\n");
  if (lcmProducts == 1 && isResultIntegerExact)
    *isResultIntegerExact = true;

  // Copy over the constraints not involving this variable.
  for (auto nbPos : nbIndices) {
    SmallVector<int64_t, 4> ineq;
    ineq.reserve(getNumCols() - 1);
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      ineq.push_back(atIneq(nbPos, l));
    }
    newFac.addInequality(ineq);
  }

  assert(newFac.getNumConstraints() ==
         lbIndices.size() * ubIndices.size() + nbIndices.size());

  // Copy over the equalities.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    SmallVector<int64_t, 4> eq;
    eq.reserve(newFac.getNumCols());
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      eq.push_back(atEq(r, l));
    }
    newFac.addEquality(eq);
  }

  // GCD tightening and normalization allows detection of more trivially
  // redundant constraints.
  newFac.gcdTightenInequalities();
  newFac.normalizeConstraintsByGCD();
  newFac.removeTrivialRedundancy();
  clearAndCopyFrom(newFac);
  LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
  LLVM_DEBUG(dump());
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "affine-structures"

void FlatAffineConstraints::projectOut(unsigned pos, unsigned num) {
  if (num == 0)
    return;

  // 'pos' can be at most getNumCols() - 2 if num > 0.
  assert((getNumCols() < 2 || pos <= getNumCols() - 2) && "invalid position");
  assert(pos + num < getNumCols() && "invalid range");

  // Eliminate as many identifiers as possible using Gaussian elimination.
  unsigned currentPos = pos;
  unsigned numToEliminate = num;
  unsigned numGaussianEliminated = 0;

  while (currentPos < getNumIds()) {
    unsigned curNumEliminated =
        gaussianEliminateIds(currentPos, currentPos + numToEliminate);
    ++currentPos;
    numToEliminate -= curNumEliminated + 1;
    numGaussianEliminated += curNumEliminated;
  }

  // Eliminate the remaining using Fourier-Motzkin.
  for (unsigned i = 0; i < num - numGaussianEliminated; i++) {
    unsigned numToEliminate = num - numGaussianEliminated - i;
    fourierMotzkinEliminate(
        getBestIdToEliminate(*this, pos, pos + numToEliminate));
  }

  // Fast/trivial simplifications.
  gcdTightenInequalities();
  // Normalize constraints after tightening since the latter impacts this, but
  // not the other way round.
  normalizeConstraintsByGCD();
}

void FlatAffineConstraints::projectOut(Value id) {
  unsigned pos;
  bool ret = findId(id, &pos);
  assert(ret);
  (void)ret;
  fourierMotzkinEliminate(pos);
}

void FlatAffineConstraints::clearConstraints() {
  equalities.resizeVertically(0);
  inequalities.resizeVertically(0);
}

namespace {

enum BoundCmpResult { Greater, Less, Equal, Unknown };

/// Compares two affine bounds whose coefficients are provided in 'first' and
/// 'second'. The last coefficient is the constant term.
static BoundCmpResult compareBounds(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  assert(a.size() == b.size());

  // For the bounds to be comparable, their corresponding identifier
  // coefficients should be equal; the constant terms are then compared to
  // determine less/greater/equal.

  if (!std::equal(a.begin(), a.end() - 1, b.begin()))
    return Unknown;

  if (a.back() == b.back())
    return Equal;

  return a.back() < b.back() ? Less : Greater;
}
} // namespace

// Returns constraints that are common to both A & B.
static void getCommonConstraints(const FlatAffineConstraints &a,
                                 const FlatAffineConstraints &b,
                                 FlatAffineConstraints &c) {
  c.reset(a.getNumDimIds(), a.getNumSymbolIds(), a.getNumLocalIds());
  // a naive O(n^2) check should be enough here given the input sizes.
  for (unsigned r = 0, e = a.getNumInequalities(); r < e; ++r) {
    for (unsigned s = 0, f = b.getNumInequalities(); s < f; ++s) {
      if (a.getInequality(r) == b.getInequality(s)) {
        c.addInequality(a.getInequality(r));
        break;
      }
    }
  }
  for (unsigned r = 0, e = a.getNumEqualities(); r < e; ++r) {
    for (unsigned s = 0, f = b.getNumEqualities(); s < f; ++s) {
      if (a.getEquality(r) == b.getEquality(s)) {
        c.addEquality(a.getEquality(r));
        break;
      }
    }
  }
}

// Computes the bounding box with respect to 'other' by finding the min of the
// lower bounds and the max of the upper bounds along each of the dimensions.
LogicalResult
FlatAffineConstraints::unionBoundingBox(const FlatAffineConstraints &otherCst) {
  assert(otherCst.getNumDimIds() == numDims && "dims mismatch");
  assert(otherCst.getIds()
             .slice(0, getNumDimIds())
             .equals(getIds().slice(0, getNumDimIds())) &&
         "dim values mismatch");
  assert(otherCst.getNumLocalIds() == 0 && "local ids not supported here");
  assert(getNumLocalIds() == 0 && "local ids not supported yet here");

  // Align `other` to this.
  Optional<FlatAffineConstraints> otherCopy;
  if (!areIdsAligned(*this, otherCst)) {
    otherCopy.emplace(FlatAffineConstraints(otherCst));
    mergeAndAlignIds(/*offset=*/numDims, this, &otherCopy.getValue());
  }

  const auto &otherAligned = otherCopy ? *otherCopy : otherCst;

  // Get the constraints common to both systems; these will be added as is to
  // the union.
  FlatAffineConstraints commonCst;
  getCommonConstraints(*this, otherAligned, commonCst);

  std::vector<SmallVector<int64_t, 8>> boundingLbs;
  std::vector<SmallVector<int64_t, 8>> boundingUbs;
  boundingLbs.reserve(2 * getNumDimIds());
  boundingUbs.reserve(2 * getNumDimIds());

  // To hold lower and upper bounds for each dimension.
  SmallVector<int64_t, 4> lb, otherLb, ub, otherUb;
  // To compute min of lower bounds and max of upper bounds for each dimension.
  SmallVector<int64_t, 4> minLb(getNumSymbolIds() + 1);
  SmallVector<int64_t, 4> maxUb(getNumSymbolIds() + 1);
  // To compute final new lower and upper bounds for the union.
  SmallVector<int64_t, 8> newLb(getNumCols()), newUb(getNumCols());

  int64_t lbFloorDivisor, otherLbFloorDivisor;
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    auto extent = getConstantBoundOnDimSize(d, &lb, &lbFloorDivisor, &ub);
    if (!extent.hasValue())
      // TODO: symbolic extents when necessary.
      // TODO: handle union if a dimension is unbounded.
      return failure();

    auto otherExtent = otherAligned.getConstantBoundOnDimSize(
        d, &otherLb, &otherLbFloorDivisor, &otherUb);
    if (!otherExtent.hasValue() || lbFloorDivisor != otherLbFloorDivisor)
      // TODO: symbolic extents when necessary.
      return failure();

    assert(lbFloorDivisor > 0 && "divisor always expected to be positive");

    auto res = compareBounds(lb, otherLb);
    // Identify min.
    if (res == BoundCmpResult::Less || res == BoundCmpResult::Equal) {
      minLb = lb;
      // Since the divisor is for a floordiv, we need to convert to ceildiv,
      // i.e., i >= expr floordiv div <=> i >= (expr - div + 1) ceildiv div <=>
      // div * i >= expr - div + 1.
      minLb.back() -= lbFloorDivisor - 1;
    } else if (res == BoundCmpResult::Greater) {
      minLb = otherLb;
      minLb.back() -= otherLbFloorDivisor - 1;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constLb = getConstantLowerBound(d);
      auto constOtherLb = otherAligned.getConstantLowerBound(d);
      if (!constLb.hasValue() || !constOtherLb.hasValue())
        return failure();
      std::fill(minLb.begin(), minLb.end(), 0);
      minLb.back() = std::min(constLb.getValue(), constOtherLb.getValue());
    }

    // Do the same for ub's but max of upper bounds. Identify max.
    auto uRes = compareBounds(ub, otherUb);
    if (uRes == BoundCmpResult::Greater || uRes == BoundCmpResult::Equal) {
      maxUb = ub;
    } else if (uRes == BoundCmpResult::Less) {
      maxUb = otherUb;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constUb = getConstantUpperBound(d);
      auto constOtherUb = otherAligned.getConstantUpperBound(d);
      if (!constUb.hasValue() || !constOtherUb.hasValue())
        return failure();
      std::fill(maxUb.begin(), maxUb.end(), 0);
      maxUb.back() = std::max(constUb.getValue(), constOtherUb.getValue());
    }

    std::fill(newLb.begin(), newLb.end(), 0);
    std::fill(newUb.begin(), newUb.end(), 0);

    // The divisor for lb, ub, otherLb, otherUb at this point is lbDivisor,
    // and so it's the divisor for newLb and newUb as well.
    newLb[d] = lbFloorDivisor;
    newUb[d] = -lbFloorDivisor;
    // Copy over the symbolic part + constant term.
    std::copy(minLb.begin(), minLb.end(), newLb.begin() + getNumDimIds());
    std::transform(newLb.begin() + getNumDimIds(), newLb.end(),
                   newLb.begin() + getNumDimIds(), std::negate<int64_t>());
    std::copy(maxUb.begin(), maxUb.end(), newUb.begin() + getNumDimIds());

    boundingLbs.push_back(newLb);
    boundingUbs.push_back(newUb);
  }

  // Clear all constraints and add the lower/upper bounds for the bounding box.
  clearConstraints();
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    addInequality(boundingLbs[d]);
    addInequality(boundingUbs[d]);
  }

  // Add the constraints that were common to both systems.
  append(commonCst);
  removeTrivialRedundancy();

  // TODO: copy over pure symbolic constraints from this and 'other' over to the
  // union (since the above are just the union along dimensions); we shouldn't
  // be discarding any other constraints on the symbols.

  return success();
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

void FlatAffineConstraints::getIneqAsAffineValueMap(
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
  getIdValues(0, pos, &operands);
  SmallVector<Value, 4> trailingOperands;
  getIdValues(pos + 1, getNumDimAndSymbolIds(), &trailingOperands);
  operands.append(trailingOperands.begin(), trailingOperands.end());
  vmap.reset(AffineMap::get(numDims - 1, numSyms, boundExpr), operands);
}

/// Returns true if the pos^th column is all zero for both inequalities and
/// equalities..
static bool isColZero(const FlatAffineConstraints &cst, unsigned pos) {
  unsigned rowPos;
  return !findConstraintWithNonZeroAt(cst, pos, /*isEq=*/false, &rowPos) &&
         !findConstraintWithNonZeroAt(cst, pos, /*isEq=*/true, &rowPos);
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
    for (unsigned i = getNumDimAndSymbolIds(), e = getNumIds(); i < e; ++i) {
      if (!memo[i] && !isColZero(*this, /*pos=*/i)) {
        LLVM_DEBUG(llvm::dbgs() << "one or more local exprs do not have an "
                                   "explicit representation");
        return IntegerSet();
      }
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

/// Find positions of inequalities and equalities that do not have a coefficient
/// for [pos, pos + num) identifiers.
static void getIndependentConstraints(const FlatAffineConstraints &cst,
                                      unsigned pos, unsigned num,
                                      SmallVectorImpl<unsigned> &nbIneqIndices,
                                      SmallVectorImpl<unsigned> &nbEqIndices) {
  assert(pos < cst.getNumIds() && "invalid start position");
  assert(pos + num <= cst.getNumIds() && "invalid limit");

  for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    unsigned c;
    for (c = pos; c < pos + num; ++c) {
      if (cst.atIneq(r, c) != 0)
        break;
    }
    if (c == pos + num)
      nbIneqIndices.push_back(r);
  }

  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    unsigned c;
    for (c = pos; c < pos + num; ++c) {
      if (cst.atEq(r, c) != 0)
        break;
    }
    if (c == pos + num)
      nbEqIndices.push_back(r);
  }
}

void FlatAffineConstraints::removeIndependentConstraints(unsigned pos,
                                                         unsigned num) {
  assert(pos + num <= getNumIds() && "invalid range");

  // Remove constraints that are independent of these identifiers.
  SmallVector<unsigned, 4> nbIneqIndices, nbEqIndices;
  getIndependentConstraints(*this, /*pos=*/0, num, nbIneqIndices, nbEqIndices);

  // Iterate in reverse so that indices don't have to be updated.
  // TODO: This method can be made more efficient (because removal of each
  // inequality leads to much shifting/copying in the underlying buffer).
  for (auto nbIndex : llvm::reverse(nbIneqIndices))
    removeInequality(nbIndex);
  for (auto nbIndex : llvm::reverse(nbEqIndices))
    removeEquality(nbIndex);
}
