//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "AffineMapDetail.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

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

} // namespace

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
  return getNumDims() >= getNumResults() &&
         *this ==
             getMinorIdentityMap(getNumDims(), getNumResults(), getContext());
}

/// Returns true if this affine map is a minor identity up to broadcasted
/// dimensions which are indicated by value 0 in the result.
bool AffineMap::isMinorIdentityWithBroadcasting(
    SmallVectorImpl<unsigned> *broadcastedDims) const {
  if (broadcastedDims)
    broadcastedDims->clear();
  if (getNumDims() < getNumResults())
    return false;
  unsigned suffixStart = getNumDims() - getNumResults();
  for (const auto &idxAndExpr : llvm::enumerate(getResults())) {
    unsigned resIdx = idxAndExpr.index();
    AffineExpr expr = idxAndExpr.value();
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      // Each result may be either a constant 0 (broadcasted dimension).
      if (constExpr.getValue() != 0)
        return false;
      if (broadcastedDims)
        broadcastedDims->push_back(resIdx);
    } else if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      // Or it may be the input dimension corresponding to this result position.
      if (dimExpr.getPosition() != suffixStart + resIdx)
        return false;
    } else {
      return false;
    }
  }
  return true;
}

/// Return true if this affine map can be converted to a minor identity with
/// broadcast by doing a permute. Return a permutation (there may be
/// several) to apply to get to a minor identity with broadcasts.
/// Ex:
///  * (d0, d1, d2) -> (0, d1) maps to minor identity (d1, 0 = d2) with
///  perm = [1, 0] and broadcast d2
///  * (d0, d1, d2) -> (d0, 0) cannot be mapped to a minor identity by
///  permutation + broadcast
///  * (d0, d1, d2, d3) -> (0, d1, d3) maps to minor identity (d1, 0 = d2, d3)
///  with perm = [1, 0, 2] and broadcast d2
///  * (d0, d1) -> (d1, 0, 0, d0) maps to minor identity (d0, d1) with extra
///  leading broadcat dimensions. The map returned would be (0, 0, d0, d1) with
///  perm = [3, 0, 1, 2]
bool AffineMap::isPermutationOfMinorIdentityWithBroadcasting(
    SmallVectorImpl<unsigned> &permutedDims) const {
  unsigned projectionStart =
      getNumResults() < getNumInputs() ? getNumInputs() - getNumResults() : 0;
  permutedDims.clear();
  SmallVector<unsigned> broadcastDims;
  permutedDims.resize(getNumResults(), 0);
  // If there are more results than input dimensions we want the new map to
  // start with broadcast dimensions in order to be a minor identity with
  // broadcasting.
  unsigned leadingBroadcast =
      getNumResults() > getNumInputs() ? getNumResults() - getNumInputs() : 0;
  llvm::SmallBitVector dimFound(std::max(getNumInputs(), getNumResults()),
                                false);
  for (const auto &idxAndExpr : llvm::enumerate(getResults())) {
    unsigned resIdx = idxAndExpr.index();
    AffineExpr expr = idxAndExpr.value();
    // Each result may be either a constant 0 (broadcast dimension) or a
    // dimension.
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      if (constExpr.getValue() != 0)
        return false;
      broadcastDims.push_back(resIdx);
    } else if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() < projectionStart)
        return false;
      unsigned newPosition =
          dimExpr.getPosition() - projectionStart + leadingBroadcast;
      permutedDims[resIdx] = newPosition;
      dimFound[newPosition] = true;
    } else {
      return false;
    }
  }
  // Find a permuation for the broadcast dimension. Since they are broadcasted
  // any valid permutation is acceptable. We just permute the dim into a slot
  // without an existing dimension.
  unsigned pos = 0;
  for (auto dim : broadcastDims) {
    while (pos < dimFound.size() && dimFound[pos]) {
      pos++;
    }
    permutedDims[dim] = pos++;
  }
  return true;
}

/// Returns an AffineMap representing a permutation.
AffineMap AffineMap::getPermutationMap(ArrayRef<unsigned> permutation,
                                       MLIRContext *context) {
  assert(!permutation.empty() &&
         "Cannot create permutation map from empty permutation vector");
  SmallVector<AffineExpr, 4> affExprs;
  for (auto index : permutation)
    affExprs.push_back(getAffineDimExpr(index, context));
  const auto *m = std::max_element(permutation.begin(), permutation.end());
  auto permutationMap = AffineMap::get(*m + 1, 0, affExprs, context);
  assert(permutationMap.isPermutation() && "Invalid permutation vector");
  return permutationMap;
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

uint64_t AffineMap::getLargestKnownDivisorOfMapExprs() {
  uint64_t gcd = 0;
  for (AffineExpr resultExpr : getResults()) {
    uint64_t thisGcd = resultExpr.getLargestKnownDivisor();
    gcd = std::gcd(gcd, thisGcd);
  }
  if (gcd == 0)
    gcd = std::numeric_limits<uint64_t>::max();
  return gcd;
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

bool AffineMap::isConstant() const {
  return llvm::all_of(getResults(), [](AffineExpr expr) {
    return expr.isa<AffineConstantExpr>();
  });
}

int64_t AffineMap::getSingleConstantResult() const {
  assert(isSingleConstant() && "map must have a single constant result");
  return getResult(0).cast<AffineConstantExpr>().getValue();
}

SmallVector<int64_t> AffineMap::getConstantResults() const {
  assert(isConstant() && "map must have only constant results");
  SmallVector<int64_t> result;
  for (auto expr : getResults())
    result.emplace_back(expr.cast<AffineConstantExpr>().getValue());
  return result;
}

unsigned AffineMap::getNumDims() const {
  assert(map && "uninitialized map storage");
  return map->numDims;
}
unsigned AffineMap::getNumSymbols() const {
  assert(map && "uninitialized map storage");
  return map->numSymbols;
}
unsigned AffineMap::getNumResults() const { return getResults().size(); }
unsigned AffineMap::getNumInputs() const {
  assert(map && "uninitialized map storage");
  return map->numDims + map->numSymbols;
}
ArrayRef<AffineExpr> AffineMap::getResults() const {
  assert(map && "uninitialized map storage");
  return map->results();
}
AffineExpr AffineMap::getResult(unsigned idx) const {
  return getResults()[idx];
}

unsigned AffineMap::getDimPosition(unsigned idx) const {
  return getResult(idx).cast<AffineDimExpr>().getPosition();
}

unsigned AffineMap::getPermutedPosition(unsigned input) const {
  assert(isPermutation() && "invalid permutation request");
  for (unsigned i = 0, numResults = getNumResults(); i < numResults; i++)
    if (getDimPosition(i) == input)
      return i;
  llvm_unreachable("incorrect permutation request");
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
void AffineMap::walkExprs(llvm::function_ref<void(AffineExpr)> callback) const {
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

/// Sparse replace method. Apply AffineExpr::replace(`expr`, `replacement`) to
/// each of the results and return a new AffineMap with the new results and
/// with the specified number of dims and symbols.
AffineMap AffineMap::replace(AffineExpr expr, AffineExpr replacement,
                             unsigned numResultDims,
                             unsigned numResultSyms) const {
  SmallVector<AffineExpr, 4> newResults;
  newResults.reserve(getNumResults());
  for (AffineExpr e : getResults())
    newResults.push_back(e.replace(expr, replacement));
  return AffineMap::get(numResultDims, numResultSyms, newResults, getContext());
}

/// Sparse replace method. Apply AffineExpr::replace(`map`) to each of the
/// results and return a new AffineMap with the new results and with the
/// specified number of dims and symbols.
AffineMap AffineMap::replace(const DenseMap<AffineExpr, AffineExpr> &map,
                             unsigned numResultDims,
                             unsigned numResultSyms) const {
  SmallVector<AffineExpr, 4> newResults;
  newResults.reserve(getNumResults());
  for (AffineExpr e : getResults())
    newResults.push_back(e.replace(map));
  return AffineMap::get(numResultDims, numResultSyms, newResults, getContext());
}

AffineMap
AffineMap::replace(const DenseMap<AffineExpr, AffineExpr> &map) const {
  SmallVector<AffineExpr, 4> newResults;
  newResults.reserve(getNumResults());
  for (AffineExpr e : getResults())
    newResults.push_back(e.replace(map));
  return AffineMap::inferFromExprList(newResults).front();
}

AffineMap AffineMap::compose(AffineMap map) const {
  assert(getNumDims() == map.getNumResults() && "Number of results mismatch");
  // Prepare `map` by concatenating the symbols and rewriting its exprs.
  unsigned numDims = map.getNumDims();
  unsigned numSymbolsThisMap = getNumSymbols();
  unsigned numSymbols = numSymbolsThisMap + map.getNumSymbols();
  SmallVector<AffineExpr, 8> newDims(numDims);
  for (unsigned idx = 0; idx < numDims; ++idx) {
    newDims[idx] = getAffineDimExpr(idx, getContext());
  }
  SmallVector<AffineExpr, 8> newSymbols(numSymbols - numSymbolsThisMap);
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

SmallVector<int64_t, 4> AffineMap::compose(ArrayRef<int64_t> values) const {
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

bool AffineMap::isProjectedPermutation(bool allowZeroInResults) const {
  if (getNumSymbols() > 0)
    return false;

  // Having more results than inputs means that results have duplicated dims or
  // zeros that can't be mapped to input dims.
  if (getNumResults() > getNumInputs())
    return false;

  SmallVector<bool, 8> seen(getNumInputs(), false);
  // A projected permutation can have, at most, only one instance of each input
  // dimension in the result expressions. Zeros are allowed as long as the
  // number of result expressions is lower or equal than the number of input
  // expressions.
  for (auto expr : getResults()) {
    if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
      if (seen[dim.getPosition()])
        return false;
      seen[dim.getPosition()] = true;
    } else {
      auto constExpr = expr.dyn_cast<AffineConstantExpr>();
      if (!allowZeroInResults || !constExpr || constExpr.getValue() != 0)
        return false;
    }
  }

  // Results are either dims or zeros and zeros can be mapped to input dims.
  return true;
}

bool AffineMap::isPermutation() const {
  if (getNumDims() != getNumResults())
    return false;
  return isProjectedPermutation();
}

AffineMap AffineMap::getSubMap(ArrayRef<unsigned> resultPos) const {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(resultPos.size());
  for (auto idx : resultPos)
    exprs.push_back(getResult(idx));
  return AffineMap::get(getNumDims(), getNumSymbols(), exprs, getContext());
}

AffineMap AffineMap::getSliceMap(unsigned start, unsigned length) const {
  return AffineMap::get(getNumDims(), getNumSymbols(),
                        getResults().slice(start, length), getContext());
}

AffineMap AffineMap::getMajorSubMap(unsigned numResults) const {
  if (numResults == 0)
    return AffineMap();
  if (numResults > getNumResults())
    return *this;
  return getSliceMap(0, numResults);
}

AffineMap AffineMap::getMinorSubMap(unsigned numResults) const {
  if (numResults == 0)
    return AffineMap();
  if (numResults > getNumResults())
    return *this;
  return getSliceMap(getNumResults() - numResults, numResults);
}

AffineMap mlir::compressDims(AffineMap map,
                             const llvm::SmallBitVector &unusedDims) {
  unsigned numDims = 0;
  SmallVector<AffineExpr> dimReplacements;
  dimReplacements.reserve(map.getNumDims());
  MLIRContext *context = map.getContext();
  for (unsigned dim = 0, e = map.getNumDims(); dim < e; ++dim) {
    if (unusedDims.test(dim))
      dimReplacements.push_back(getAffineConstantExpr(0, context));
    else
      dimReplacements.push_back(getAffineDimExpr(numDims++, context));
  }
  SmallVector<AffineExpr> resultExprs;
  resultExprs.reserve(map.getNumResults());
  for (auto e : map.getResults())
    resultExprs.push_back(e.replaceDims(dimReplacements));
  return AffineMap::get(numDims, map.getNumSymbols(), resultExprs, context);
}

AffineMap mlir::compressUnusedDims(AffineMap map) {
  return compressDims(map, getUnusedDimsBitVector({map}));
}

static SmallVector<AffineMap>
compressUnusedImpl(ArrayRef<AffineMap> maps,
                   llvm::function_ref<AffineMap(AffineMap)> compressionFun) {
  if (maps.empty())
    return SmallVector<AffineMap>();
  SmallVector<AffineExpr> allExprs;
  allExprs.reserve(maps.size() * maps.front().getNumResults());
  unsigned numDims = maps.front().getNumDims(),
           numSymbols = maps.front().getNumSymbols();
  for (auto m : maps) {
    assert(numDims == m.getNumDims() && numSymbols == m.getNumSymbols() &&
           "expected maps with same num dims and symbols");
    llvm::append_range(allExprs, m.getResults());
  }
  AffineMap unifiedMap = compressionFun(
      AffineMap::get(numDims, numSymbols, allExprs, maps.front().getContext()));
  unsigned unifiedNumDims = unifiedMap.getNumDims(),
           unifiedNumSymbols = unifiedMap.getNumSymbols();
  ArrayRef<AffineExpr> unifiedResults = unifiedMap.getResults();
  SmallVector<AffineMap> res;
  res.reserve(maps.size());
  for (auto m : maps) {
    res.push_back(AffineMap::get(unifiedNumDims, unifiedNumSymbols,
                                 unifiedResults.take_front(m.getNumResults()),
                                 m.getContext()));
    unifiedResults = unifiedResults.drop_front(m.getNumResults());
  }
  return res;
}

SmallVector<AffineMap> mlir::compressUnusedDims(ArrayRef<AffineMap> maps) {
  return compressUnusedImpl(maps,
                            [](AffineMap m) { return compressUnusedDims(m); });
}

AffineMap mlir::compressSymbols(AffineMap map,
                                const llvm::SmallBitVector &unusedSymbols) {
  unsigned numSymbols = 0;
  SmallVector<AffineExpr> symReplacements;
  symReplacements.reserve(map.getNumSymbols());
  MLIRContext *context = map.getContext();
  for (unsigned sym = 0, e = map.getNumSymbols(); sym < e; ++sym) {
    if (unusedSymbols.test(sym))
      symReplacements.push_back(getAffineConstantExpr(0, context));
    else
      symReplacements.push_back(getAffineSymbolExpr(numSymbols++, context));
  }
  SmallVector<AffineExpr> resultExprs;
  resultExprs.reserve(map.getNumResults());
  for (auto e : map.getResults())
    resultExprs.push_back(e.replaceSymbols(symReplacements));
  return AffineMap::get(map.getNumDims(), numSymbols, resultExprs, context);
}

AffineMap mlir::compressUnusedSymbols(AffineMap map) {
  llvm::SmallBitVector unusedSymbols(map.getNumSymbols(), true);
  map.walkExprs([&](AffineExpr expr) {
    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
      unusedSymbols.reset(symExpr.getPosition());
  });
  return compressSymbols(map, unusedSymbols);
}

SmallVector<AffineMap> mlir::compressUnusedSymbols(ArrayRef<AffineMap> maps) {
  return compressUnusedImpl(
      maps, [](AffineMap m) { return compressUnusedSymbols(m); });
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
  for (const auto &en : llvm::enumerate(map.getResults())) {
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

AffineMap mlir::inverseAndBroadcastProjectedPermutation(AffineMap map) {
  assert(map.isProjectedPermutation(/*allowZeroInResults=*/true));
  MLIRContext *context = map.getContext();
  AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  // Start with all the results as 0.
  SmallVector<AffineExpr, 4> exprs(map.getNumInputs(), zero);
  for (unsigned i : llvm::seq(unsigned(0), map.getNumResults())) {
    // Skip zeros from input map. 'exprs' is already initialized to zero.
    if (auto constExpr = map.getResult(i).dyn_cast<AffineConstantExpr>()) {
      assert(constExpr.getValue() == 0 &&
             "Unexpected constant in projected permutation");
      (void)constExpr;
      continue;
    }

    // Reverse each dimension existing in the original map result.
    exprs[map.getDimPosition(i)] = getAffineDimExpr(i, context);
  }
  return AffineMap::get(map.getNumResults(), /*symbolCount=*/0, exprs, context);
}

AffineMap mlir::concatAffineMaps(ArrayRef<AffineMap> maps) {
  unsigned numResults = 0, numDims = 0, numSymbols = 0;
  for (auto m : maps)
    numResults += m.getNumResults();
  SmallVector<AffineExpr, 8> results;
  results.reserve(numResults);
  for (auto m : maps) {
    for (auto res : m.getResults())
      results.push_back(res.shiftSymbols(m.getNumSymbols(), numSymbols));

    numSymbols += m.getNumSymbols();
    numDims = std::max(m.getNumDims(), numDims);
  }
  return AffineMap::get(numDims, numSymbols, results,
                        maps.front().getContext());
}

AffineMap mlir::getProjectedMap(AffineMap map,
                                const llvm::SmallBitVector &unusedDims) {
  return compressUnusedSymbols(compressDims(map, unusedDims));
}

llvm::SmallBitVector mlir::getUnusedDimsBitVector(ArrayRef<AffineMap> maps) {
  unsigned numDims = maps[0].getNumDims();
  llvm::SmallBitVector numDimsBitVector(numDims, true);
  for (const auto &m : maps) {
    for (unsigned i = 0; i < numDims; ++i) {
      if (m.isFunctionOfDim(i))
        numDimsBitVector.reset(i);
    }
  }
  return numDimsBitVector;
}

//===----------------------------------------------------------------------===//
// MutableAffineMap.
//===----------------------------------------------------------------------===//

MutableAffineMap::MutableAffineMap(AffineMap map)
    : results(map.getResults().begin(), map.getResults().end()),
      numDims(map.getNumDims()), numSymbols(map.getNumSymbols()),
      context(map.getContext()) {}

void MutableAffineMap::reset(AffineMap map) {
  results.clear();
  numDims = map.getNumDims();
  numSymbols = map.getNumSymbols();
  context = map.getContext();
  llvm::append_range(results, map.getResults());
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
