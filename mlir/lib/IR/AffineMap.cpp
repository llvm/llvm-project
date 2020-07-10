//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "AffineMapDetail.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// AffineExprConstantFolder evaluates an affine expression using constant
// operands passed in 'operandConsts'. Returns an IntegerAttr attribute
// representing the constant value of the affine expression evaluated on
// constant 'operandConsts', or nullptr if it can't be folded.
class AffineExprConstantFolder {
public:
  AffineExprConstantFolder(unsigned numDims, ArrayRef<Attribute> operandConsts)
      : numDims(numDims), operandConsts(operandConsts) {}

  /// Attempt to constant fold the specified affine expr, or return null on
  /// failure.
  IntegerAttr constantFold(AffineExpr expr) {
    if (auto result = constantFoldImpl(expr))
      return IntegerAttr::get(IndexType::get(expr.getContext()), *result);
    return nullptr;
  }

private:
  Optional<int64_t> constantFoldImpl(AffineExpr expr) {
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
    case AffineExprKind::Mul:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
    case AffineExprKind::Mod:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return mod(lhs, rhs); });
    case AffineExprKind::FloorDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return floorDiv(lhs, rhs); });
    case AffineExprKind::CeilDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return ceilDiv(lhs, rhs); });
    case AffineExprKind::Constant:
      return expr.cast<AffineConstantExpr>().getValue();
    case AffineExprKind::DimId:
      if (auto attr = operandConsts[expr.cast<AffineDimExpr>().getPosition()]
                          .dyn_cast_or_null<IntegerAttr>())
        return attr.getInt();
      return llvm::None;
    case AffineExprKind::SymbolId:
      if (auto attr = operandConsts[numDims +
                                    expr.cast<AffineSymbolExpr>().getPosition()]
                          .dyn_cast_or_null<IntegerAttr>())
        return attr.getInt();
      return llvm::None;
    }
    llvm_unreachable("Unknown AffineExpr");
  }

  // TODO: Change these to operate on APInts too.
  Optional<int64_t> constantFoldBinExpr(AffineExpr expr,
                                        int64_t (*op)(int64_t, int64_t)) {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    if (auto lhs = constantFoldImpl(binOpExpr.getLHS()))
      if (auto rhs = constantFoldImpl(binOpExpr.getRHS()))
        return op(*lhs, *rhs);
    return llvm::None;
  }

  // The number of dimension operands in AffineMap containing this expression.
  unsigned numDims;
  // The constant valued operands used to evaluate this AffineExpr.
  ArrayRef<Attribute> operandConsts;
};

} // end anonymous namespace

/// Returns a single constant result affine map.
AffineMap AffineMap::getConstantMap(int64_t val, MLIRContext *context) {
  return get(/*dimCount=*/0, /*symbolCount=*/0,
             {getAffineConstantExpr(val, context)});
}

/// Returns an identity affine map (d0, ..., dn) -> (dp, ..., dn) on the most
/// minor dimensions.
AffineMap AffineMap::getMinorIdentityMap(unsigned dims, unsigned results,
                                         MLIRContext *context) {
  assert(dims >= results && "Dimension mismatch");
  auto id = AffineMap::getMultiDimIdentityMap(dims, context);
  return AffineMap::get(dims, 0, id.getResults().take_back(results), context);
}

bool AffineMap::isMinorIdentity() const {
  return *this ==
         getMinorIdentityMap(getNumDims(), getNumResults(), getContext());
}

/// Returns an AffineMap representing a permutation.
AffineMap AffineMap::getPermutationMap(ArrayRef<unsigned> permutation,
                                       MLIRContext *context) {
  assert(!permutation.empty() &&
         "Cannot create permutation map from empty permutation vector");
  SmallVector<AffineExpr, 4> affExprs;
  for (auto index : permutation)
    affExprs.push_back(getAffineDimExpr(index, context));
  auto m = std::max_element(permutation.begin(), permutation.end());
  auto permutationMap = AffineMap::get(*m + 1, 0, affExprs, context);
  assert(permutationMap.isPermutation() && "Invalid permutation vector");
  return permutationMap;
}

template <typename AffineExprContainer>
static void getMaxDimAndSymbol(ArrayRef<AffineExprContainer> exprsList,
                               int64_t &maxDim, int64_t &maxSym) {
  for (const auto &exprs : exprsList) {
    for (auto expr : exprs) {
      expr.walk([&maxDim, &maxSym](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineDimExpr>())
          maxDim = std::max(maxDim, static_cast<int64_t>(d.getPosition()));
        if (auto s = e.dyn_cast<AffineSymbolExpr>())
          maxSym = std::max(maxSym, static_cast<int64_t>(s.getPosition()));
      });
    }
  }
}

template <typename AffineExprContainer>
static SmallVector<AffineMap, 4>
inferFromExprList(ArrayRef<AffineExprContainer> exprsList) {
  assert(!exprsList.empty());
  assert(!exprsList[0].empty());
  auto context = exprsList[0][0].getContext();
  int64_t maxDim = -1, maxSym = -1;
  getMaxDimAndSymbol(exprsList, maxDim, maxSym);
  SmallVector<AffineMap, 4> maps;
  maps.reserve(exprsList.size());
  for (const auto &exprs : exprsList)
    maps.push_back(AffineMap::get(/*dimCount=*/maxDim + 1,
                                  /*symbolCount=*/maxSym + 1, exprs, context));
  return maps;
}

SmallVector<AffineMap, 4>
AffineMap::inferFromExprList(ArrayRef<ArrayRef<AffineExpr>> exprsList) {
  return ::inferFromExprList(exprsList);
}

SmallVector<AffineMap, 4>
AffineMap::inferFromExprList(ArrayRef<SmallVector<AffineExpr, 4>> exprsList) {
  return ::inferFromExprList(exprsList);
}

AffineMap AffineMap::getMultiDimIdentityMap(unsigned numDims,
                                            MLIRContext *context) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i)
    dimExprs.push_back(mlir::getAffineDimExpr(i, context));
  return get(/*dimCount=*/numDims, /*symbolCount=*/0, dimExprs, context);
}

MLIRContext *AffineMap::getContext() const { return map->context; }

bool AffineMap::isIdentity() const {
  if (getNumDims() != getNumResults())
    return false;
  ArrayRef<AffineExpr> results = getResults();
  for (unsigned i = 0, numDims = getNumDims(); i < numDims; ++i) {
    auto expr = results[i].dyn_cast<AffineDimExpr>();
    if (!expr || expr.getPosition() != i)
      return false;
  }
  return true;
}

bool AffineMap::isEmpty() const {
  return getNumDims() == 0 && getNumSymbols() == 0 && getNumResults() == 0;
}

bool AffineMap::isSingleConstant() const {
  return getNumResults() == 1 && getResult(0).isa<AffineConstantExpr>();
}

int64_t AffineMap::getSingleConstantResult() const {
  assert(isSingleConstant() && "map must have a single constant result");
  return getResult(0).cast<AffineConstantExpr>().getValue();
}

unsigned AffineMap::getNumDims() const {
  assert(map && "uninitialized map storage");
  return map->numDims;
}
unsigned AffineMap::getNumSymbols() const {
  assert(map && "uninitialized map storage");
  return map->numSymbols;
}
unsigned AffineMap::getNumResults() const {
  assert(map && "uninitialized map storage");
  return map->results.size();
}
unsigned AffineMap::getNumInputs() const {
  assert(map && "uninitialized map storage");
  return map->numDims + map->numSymbols;
}

ArrayRef<AffineExpr> AffineMap::getResults() const {
  assert(map && "uninitialized map storage");
  return map->results;
}
AffineExpr AffineMap::getResult(unsigned idx) const {
  assert(map && "uninitialized map storage");
  return map->results[idx];
}

/// Folds the results of the application of an affine map on the provided
/// operands to a constant if possible. Returns false if the folding happens,
/// true otherwise.
LogicalResult
AffineMap::constantFold(ArrayRef<Attribute> operandConstants,
                        SmallVectorImpl<Attribute> &results) const {
  // Attempt partial folding.
  SmallVector<int64_t, 2> integers;
  partialConstantFold(operandConstants, &integers);

  // If all expressions folded to a constant, populate results with attributes
  // containing those constants.
  if (integers.empty())
    return failure();

  auto range = llvm::map_range(integers, [this](int64_t i) {
    return IntegerAttr::get(IndexType::get(getContext()), i);
  });
  results.append(range.begin(), range.end());
  return success();
}

AffineMap
AffineMap::partialConstantFold(ArrayRef<Attribute> operandConstants,
                               SmallVectorImpl<int64_t> *results) const {
  assert(getNumInputs() == operandConstants.size());

  // Fold each of the result expressions.
  AffineExprConstantFolder exprFolder(getNumDims(), operandConstants);
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(getNumResults());

  for (auto expr : getResults()) {
    auto folded = exprFolder.constantFold(expr);
    // If did not fold to a constant, keep the original expression, and clear
    // the integer results vector.
    if (folded) {
      exprs.push_back(
          getAffineConstantExpr(folded.getInt(), folded.getContext()));
      if (results)
        results->push_back(folded.getInt());
    } else {
      exprs.push_back(expr);
      if (results) {
        results->clear();
        results = nullptr;
      }
    }
  }

  return get(getNumDims(), getNumSymbols(), exprs, getContext());
}

/// Walk all of the AffineExpr's in this mapping. Each node in an expression
/// tree is visited in postorder.
void AffineMap::walkExprs(std::function<void(AffineExpr)> callback) const {
  for (auto expr : getResults())
    expr.walk(callback);
}

/// This method substitutes any uses of dimensions and symbols (e.g.
/// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
/// expression mapping.  Because this can be used to eliminate dims and
/// symbols, the client needs to specify the number of dims and symbols in
/// the result.  The returned map always has the same number of results.
AffineMap AffineMap::replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                           ArrayRef<AffineExpr> symReplacements,
                                           unsigned numResultDims,
                                           unsigned numResultSyms) const {
  SmallVector<AffineExpr, 8> results;
  results.reserve(getNumResults());
  for (auto expr : getResults())
    results.push_back(
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements));

  return get(numResultDims, numResultSyms, results, getContext());
}

AffineMap AffineMap::compose(AffineMap map) {
  assert(getNumDims() == map.getNumResults() && "Number of results mismatch");
  // Prepare `map` by concatenating the symbols and rewriting its exprs.
  unsigned numDims = map.getNumDims();
  unsigned numSymbolsThisMap = getNumSymbols();
  unsigned numSymbols = numSymbolsThisMap + map.getNumSymbols();
  SmallVector<AffineExpr, 8> newDims(numDims);
  for (unsigned idx = 0; idx < numDims; ++idx) {
    newDims[idx] = getAffineDimExpr(idx, getContext());
  }
  SmallVector<AffineExpr, 8> newSymbols(numSymbols);
  for (unsigned idx = numSymbolsThisMap; idx < numSymbols; ++idx) {
    newSymbols[idx - numSymbolsThisMap] =
        getAffineSymbolExpr(idx, getContext());
  }
  auto newMap =
      map.replaceDimsAndSymbols(newDims, newSymbols, numDims, numSymbols);
  SmallVector<AffineExpr, 8> exprs;
  exprs.reserve(getResults().size());
  for (auto expr : getResults())
    exprs.push_back(expr.compose(newMap));
  return AffineMap::get(numDims, numSymbols, exprs, map.getContext());
}

SmallVector<int64_t, 4> AffineMap::compose(ArrayRef<int64_t> values) {
  assert(getNumSymbols() == 0 && "Expected symbol-less map");
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(values.size());
  MLIRContext *ctx = getContext();
  for (auto v : values)
    exprs.push_back(getAffineConstantExpr(v, ctx));
  auto resMap = compose(AffineMap::get(0, 0, exprs, ctx));
  SmallVector<int64_t, 4> res;
  res.reserve(resMap.getNumResults());
  for (auto e : resMap.getResults())
    res.push_back(e.cast<AffineConstantExpr>().getValue());
  return res;
}

bool AffineMap::isProjectedPermutation() {
  if (getNumSymbols() > 0)
    return false;
  SmallVector<bool, 8> seen(getNumInputs(), false);
  for (auto expr : getResults()) {
    if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
      if (seen[dim.getPosition()])
        return false;
      seen[dim.getPosition()] = true;
      continue;
    }
    return false;
  }
  return true;
}

bool AffineMap::isPermutation() {
  if (getNumDims() != getNumResults())
    return false;
  return isProjectedPermutation();
}

AffineMap AffineMap::getSubMap(ArrayRef<unsigned> resultPos) {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(resultPos.size());
  for (auto idx : resultPos)
    exprs.push_back(getResult(idx));
  return AffineMap::get(getNumDims(), getNumSymbols(), exprs, getContext());
}

AffineMap AffineMap::getMajorSubMap(unsigned numResults) {
  if (numResults == 0)
    return AffineMap();
  if (numResults > getNumResults())
    return *this;
  return getSubMap(llvm::to_vector<4>(llvm::seq<unsigned>(0, numResults)));
}

AffineMap AffineMap::getMinorSubMap(unsigned numResults) {
  if (numResults == 0)
    return AffineMap();
  if (numResults > getNumResults())
    return *this;
  return getSubMap(llvm::to_vector<4>(
      llvm::seq<unsigned>(getNumResults() - numResults, getNumResults())));
}

AffineMap mlir::simplifyAffineMap(AffineMap map) {
  SmallVector<AffineExpr, 8> exprs;
  for (auto e : map.getResults()) {
    exprs.push_back(
        simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols()));
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs,
                        map.getContext());
}

AffineMap mlir::removeDuplicateExprs(AffineMap map) {
  auto results = map.getResults();
  SmallVector<AffineExpr, 4> uniqueExprs(results.begin(), results.end());
  uniqueExprs.erase(std::unique(uniqueExprs.begin(), uniqueExprs.end()),
                    uniqueExprs.end());
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), uniqueExprs,
                        map.getContext());
}

AffineMap mlir::inversePermutation(AffineMap map) {
  if (map.isEmpty())
    return map;
  assert(map.getNumSymbols() == 0 && "expected map without symbols");
  SmallVector<AffineExpr, 4> exprs(map.getNumDims());
  for (auto en : llvm::enumerate(map.getResults())) {
    auto expr = en.value();
    // Skip non-permutations.
    if (auto d = expr.dyn_cast<AffineDimExpr>()) {
      if (exprs[d.getPosition()])
        continue;
      exprs[d.getPosition()] = getAffineDimExpr(en.index(), d.getContext());
    }
  }
  SmallVector<AffineExpr, 4> seenExprs;
  seenExprs.reserve(map.getNumDims());
  for (auto expr : exprs)
    if (expr)
      seenExprs.push_back(expr);
  if (seenExprs.size() != map.getNumInputs())
    return AffineMap();
  return AffineMap::get(map.getNumResults(), 0, seenExprs, map.getContext());
}

AffineMap mlir::concatAffineMaps(ArrayRef<AffineMap> maps) {
  unsigned numResults = 0;
  for (auto m : maps)
    numResults += m.getNumResults();
  unsigned numDims = 0;
  SmallVector<AffineExpr, 8> results;
  results.reserve(numResults);
  for (auto m : maps) {
    assert(m.getNumSymbols() == 0 && "expected map without symbols");
    results.append(m.getResults().begin(), m.getResults().end());
    numDims = std::max(m.getNumDims(), numDims);
  }
  return AffineMap::get(numDims, /*numSymbols=*/0, results,
                        maps.front().getContext());
}

//===----------------------------------------------------------------------===//
// MutableAffineMap.
//===----------------------------------------------------------------------===//

MutableAffineMap::MutableAffineMap(AffineMap map)
    : numDims(map.getNumDims()), numSymbols(map.getNumSymbols()),
      context(map.getContext()) {
  for (auto result : map.getResults())
    results.push_back(result);
}

void MutableAffineMap::reset(AffineMap map) {
  results.clear();
  numDims = map.getNumDims();
  numSymbols = map.getNumSymbols();
  context = map.getContext();
  for (auto result : map.getResults())
    results.push_back(result);
}

bool MutableAffineMap::isMultipleOf(unsigned idx, int64_t factor) const {
  if (results[idx].isMultipleOf(factor))
    return true;

  // TODO: use simplifyAffineExpr and FlatAffineConstraints to
  // complete this (for a more powerful analysis).
  return false;
}

// Simplifies the result affine expressions of this map. The expressions have to
// be pure for the simplification implemented.
void MutableAffineMap::simplify() {
  // Simplify each of the results if possible.
  // TODO: functional-style map
  for (unsigned i = 0, e = getNumResults(); i < e; i++) {
    results[i] = simplifyAffineExpr(getResult(i), numDims, numSymbols);
  }
}

AffineMap MutableAffineMap::getAffineMap() const {
  return AffineMap::get(numDims, numSymbols, results, context);
}
