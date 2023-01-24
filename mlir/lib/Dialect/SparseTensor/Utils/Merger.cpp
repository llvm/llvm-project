//===- Merger.cpp - Implementation of iteration lattices ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace mlir {
namespace sparse_tensor {

enum class ExpArity {
  kNullary,
  kUnary,
  kBinary,
};

static ExpArity getExpArity(TensorExp::Kind k) {
  switch (k) {
  // Leaf.
  case TensorExp::Kind::kTensor:
  case TensorExp::Kind::kInvariant:
  case TensorExp::Kind::kLoopVar:
    return ExpArity::kNullary;
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
  case TensorExp::Kind::kCeilF:
  case TensorExp::Kind::kFloorF:
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kCIm:
  case TensorExp::Kind::kCRe:
  case TensorExp::Kind::kBitCast:
  case TensorExp::Kind::kBinaryBranch:
  case TensorExp::Kind::kUnary:
  case TensorExp::Kind::kSelect:
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
    return ExpArity::kUnary;
  // Binary operations.
  case TensorExp::Kind::kDivF:
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
  case TensorExp::Kind::kShrS:
  case TensorExp::Kind::kShrU:
  case TensorExp::Kind::kShlI:
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
  case TensorExp::Kind::kAndI:
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
  case TensorExp::Kind::kOrI:
  case TensorExp::Kind::kXorI:
  case TensorExp::Kind::kBinary:
  case TensorExp::Kind::kReduce:
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
    return ExpArity::kBinary;
  }
  llvm_unreachable("unexpected kind");
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

TensorExp::TensorExp(TensorExp::Kind k, unsigned x, ExprId y, Value v,
                     Operation *o)
    : kind(k), val(v), op(o) {
  switch (kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
    assert(x != detail::kInvalidId && y == detail::kInvalidId && !v && !o);
    tensor = x;
    return;
  case TensorExp::Kind::kInvariant:
    assert(x == detail::kInvalidId && y == detail::kInvalidId && v && !o);
    return;
  case TensorExp::Kind::kLoopVar:
    assert(x != detail::kInvalidId && y == detail::kInvalidId && !v && !o);
    loop = x;
    return;
  // Unary operations.
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
  case TensorExp::Kind::kCeilF:
  case TensorExp::Kind::kFloorF:
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
  case TensorExp::Kind::kCIm:
  case TensorExp::Kind::kCRe:
    assert(x != detail::kInvalidId && y == detail::kInvalidId && !v && !o);
    children.e0 = x;
    children.e1 = y;
    return;
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kBitCast:
    assert(x != detail::kInvalidId && y == detail::kInvalidId && v && !o);
    children.e0 = x;
    children.e1 = y;
    return;
  case TensorExp::Kind::kBinaryBranch:
  case TensorExp::Kind::kSelect:
    assert(x != detail::kInvalidId && y == detail::kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    return;
  case TensorExp::Kind::kUnary:
    // No assertion on y can be made, as the branching paths involve both
    // a unary (`mapSet`) and binary (`disjSet`) pathway.
    assert(x != detail::kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    return;
  // Binary operations.
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
  case TensorExp::Kind::kDivF:
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
  case TensorExp::Kind::kAndI:
  case TensorExp::Kind::kOrI:
  case TensorExp::Kind::kXorI:
  case TensorExp::Kind::kShrS:
  case TensorExp::Kind::kShrU:
  case TensorExp::Kind::kShlI:
    assert(x != detail::kInvalidId && y != detail::kInvalidId && !v && !o);
    children.e0 = x;
    children.e1 = y;
    return;
  case TensorExp::Kind::kBinary:
  case TensorExp::Kind::kReduce:
    assert(x != detail::kInvalidId && y != detail::kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    return;
  }
  llvm_unreachable("unexpected kind");
}

Merger::Merger(unsigned numInputOutputTensors, unsigned numNativeLoops,
               unsigned numFilterLoops, unsigned maxLvlRank)
    : outTensor(numInputOutputTensors - 1),
      syntheticTensor(numInputOutputTensors),
      numTensors(numInputOutputTensors + 1), numNativeLoops(numNativeLoops),
      numLoops(numNativeLoops + numFilterLoops), hasSparseOut(false),
      lvlTypes(numTensors,
               std::vector<DimLevelType>(numLoops, DimLevelType::Undef)),
      loopToLvl(numTensors,
                std::vector<std::optional<Level>>(numLoops, std::nullopt)),
      lvlToLoop(numTensors,
                std::vector<std::optional<LoopId>>(maxLvlRank, std::nullopt)),
      loopToDependencies(
          numLoops, std::vector<std::optional<std::pair<Level, DimLevelType>>>(
                        numTensors, std::nullopt)),
      levelToDependentLoop(numTensors, std::vector<std::vector<LoopId>>(
                                           maxLvlRank, std::vector<LoopId>())),
      loopBounds(numLoops, std::make_pair(numTensors, numLoops)) {}

//===----------------------------------------------------------------------===//
// Lattice methods.
//===----------------------------------------------------------------------===//

ExprId Merger::addTensorExp(TensorId t) {
  assert(isValidTensorId(t));
  const ExprId eNew(tensorExps.size());
  tensorExps.emplace_back(TensorExp::Kind::kTensor, t, detail::kInvalidId,
                          Value(), nullptr);
  return eNew;
}

ExprId Merger::addLoopVarExp(LoopId i) {
  assert(isValidLoopId(i));
  const ExprId eNew(tensorExps.size());
  tensorExps.emplace_back(TensorExp::Kind::kLoopVar, i, detail::kInvalidId,
                          Value(), nullptr);
  return eNew;
}

ExprId Merger::addInvariantExp(Value v) {
  const ExprId eNew(tensorExps.size());
  tensorExps.emplace_back(TensorExp::Kind::kInvariant, detail::kInvalidId,
                          detail::kInvalidId, v, nullptr);
  return eNew;
}

ExprId Merger::addExp(TensorExp::Kind k, ExprId e0, ExprId e1, Operation *op) {
  assert(k > TensorExp::Kind::kLoopVar);
  const ExprId eNew(tensorExps.size());
  tensorExps.emplace_back(k, e0, e1, Value(), op);
  return eNew;
}

ExprId Merger::addExp(TensorExp::Kind k, ExprId e, Value v, Operation *op) {
  assert(k > TensorExp::Kind::kLoopVar);
  const ExprId eNew(tensorExps.size());
  tensorExps.emplace_back(k, e, detail::kInvalidId, v, op);
  return eNew;
}

LatPointId Merger::addLat(TensorId t, LoopId i, ExprId e) {
  const LatPointId pNew(latPoints.size());
  const unsigned size = numLoops * numTensors;
  const TensorLoopId b = makeTensorLoopId(t, i);
  latPoints.emplace_back(size, e);
  latPoints[pNew].bits.set(b);
  return pNew;
}

LatPointId Merger::addLat(const BitVector &bits, ExprId e) {
  assert(bits.size() == numLoops * numTensors);
  const LatPointId pNew(latPoints.size());
  latPoints.emplace_back(bits, e);
  return pNew;
}

LatSetId Merger::addSet() {
  const LatSetId sNew(latSets.size());
  latSets.emplace_back();
  return sNew;
}

LatPointId Merger::conjLat(TensorExp::Kind kind, LatPointId p0, LatPointId p1,
                           Operation *op) {
  const LatPointId pNew(latPoints.size());
  const auto &point0 = lat(p0);
  const auto &point1 = lat(p1);
  BitVector bits(point0.bits);
  bits |= point1.bits;
  const ExprId e = addExp(kind, point0.exp, point1.exp, op);
  latPoints.emplace_back(bits, e);
  return pNew;
}

LatSetId Merger::conjSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                         Operation *op) {
  const LatSetId sNew = addSet();
  auto &setNew = latSets[sNew];
  for (const LatPointId p0 : set(s0))
    for (const LatPointId p1 : set(s1))
      setNew.push_back(conjLat(kind, p0, p1, op));
  return sNew;
}

LatSetId Merger::disjSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                         Operation *op) {
  const LatSetId sNew = conjSet(kind, s0, s1, op);
  // Followed by all in s0.
  latSets[sNew].append(latSets[s0]);
  // Map binary 0-y to unary -y.
  // TODO: move this if-else logic into buildLattices
  if (kind == TensorExp::Kind::kSubF)
    s1 = mapSet(TensorExp::Kind::kNegF, s1);
  else if (kind == TensorExp::Kind::kSubC)
    s1 = mapSet(TensorExp::Kind::kNegC, s1);
  else if (kind == TensorExp::Kind::kSubI)
    s1 = mapSet(TensorExp::Kind::kNegI, s1);
  // Followed by all in s1.
  latSets[sNew].append(latSets[s1]);
  return sNew;
}

LatSetId Merger::combiSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                          Operation *orig, bool includeLeft,
                          TensorExp::Kind ltrans, Operation *opleft,
                          bool includeRight, TensorExp::Kind rtrans,
                          Operation *opright) {
  const LatSetId sNew = conjSet(kind, s0, s1, orig);
  // Left Region.
  if (includeLeft) {
    if (opleft)
      s0 = mapSet(ltrans, s0, Value(), opleft);
    latSets[sNew].append(latSets[s0]);
  }
  // Right Region.
  if (includeRight) {
    if (opright)
      s1 = mapSet(rtrans, s1, Value(), opright);
    latSets[sNew].append(latSets[s1]);
  }
  return sNew;
}

LatSetId Merger::mapSet(TensorExp::Kind kind, LatSetId s0, Value v,
                        Operation *op) {
  assert(TensorExp::Kind::kAbsF <= kind && kind <= TensorExp::Kind::kSelect);
  const LatSetId sNew = addSet();
  auto &setNew = latSets[sNew];
  for (const LatPointId p : set(s0)) {
    const auto &point = latPoints[p];
    setNew.push_back(addLat(point.bits, addExp(kind, point.exp, v, op)));
  }
  return sNew;
}

LatSetId Merger::optimizeSet(LatSetId s0) {
  const LatSetId sNew = addSet();
  auto &setNew = latSets[sNew];
  const auto &set0 = set(s0);
  assert(!set0.empty());
  const LatPointId p0 = set0[0];
  for (const LatPointId p1 : set0) {
    bool add = true;
    if (p0 != p1) {
      // Check whether this is a straightforward copy.
      if (expIsTensor(latPoints[p1].exp, outTensor))
        continue;
      // Check whether this conjunction is already covered.
      for (const LatPointId p2 : setNew) {
        assert(!latGT(p1, p2)); // Lj => Li would be bad
        if (onlyDenseDiff(p2, p1)) {
          add = false;
          break;
        }
      }
      assert(!add || latGT(p0, p1));
    }
    if (add)
      setNew.push_back(p1);
  }
  for (const LatPointId p : setNew)
    latPoints[p].simple = simplifyCond(sNew, p);
  return sNew;
}

BitVector Merger::simplifyCond(LatSetId s0, LatPointId p0) {
  // First determine if this lattice point is a *singleton*, i.e.,
  // the last point in a lattice, no other is less than this one.
  bool isSingleton = true;
  for (const LatPointId p1 : set(s0)) {
    if (p0 != p1 && latGT(p0, p1)) {
      isSingleton = false;
      break;
    }
  }

  BitVector simple(latPoints[p0].bits);
  bool reset = isSingleton && hasAnySparse(simple);
  const TensorLoopId be = simple.size();
  TensorLoopId offset = 0; // relative to the end
  if (!reset)
    // Starts resetting from a dense level, so that the first bit (if kept)
    // is not undefined level-type.
    for (unsigned b = 0; b < be; b++) {
      if (simple[b] && isDenseDLT(getDimLevelType(TensorLoopId{b}))) {
        offset = be - b - 1; // relative to the end
        break;
      }
    }

  // Now apply the two basic rules. We also iterate the bits reversely to always
  // keep the rightmost bit (which could possibly be a synthetic tensor).
  for (unsigned b = be - 1 - offset, i = 0; i < be;
       b = b == 0 ? be - 1 : b - 1, i++) {
    // Slice on dense level has `locate` property as well, and can be optimized.
    if (simple[b] && !isSparseLvlWithNonTrivialIdxExp(b)) {
      const auto dlt = getDimLevelType(b);
      if (!isCompressedDLT(dlt) && !isSingletonDLT(dlt)) {
        if (reset)
          simple.reset(b);
        reset = true;
      }
    }
  }
  return simple;
}

bool Merger::latGT(LatPointId i, LatPointId j) const {
  const BitVector &bitsi = lat(i).bits;
  const BitVector &bitsj = lat(j).bits;
  assert(bitsi.size() == bitsj.size());
  if (bitsi.count() > bitsj.count()) {
    for (TensorLoopId b = 0, be = bitsj.size(); b < be; b++)
      if (bitsj[b] && !bitsi[b])
        return false;
    return true;
  }
  return false;
}

bool Merger::onlyDenseDiff(LatPointId i, LatPointId j) const {
  BitVector tmp(latPoints[j].bits);
  tmp ^= latPoints[i].bits;
  return !hasAnySparse(tmp);
}

bool Merger::expContainsTensor(ExprId e, TensorId t) const {
  const auto &expr = exp(e);
  // First we check `expIsTensor`.
  if (expr.kind == TensorExp::Kind::kTensor)
    return expr.tensor == t;

  switch (getExpArity(expr.kind)) {
  case ExpArity::kNullary:
    return false;
  case ExpArity::kUnary: {
    const ExprId e0 = expr.children.e0;
    return expContainsTensor(e0, t);
  }
  case ExpArity::kBinary: {
    const ExprId e0 = expr.children.e0;
    const ExprId e1 = expr.children.e1;
    return expContainsTensor(e0, t) || expContainsTensor(e1, t);
  }
  }
  llvm_unreachable("unexpected arity");
}

bool Merger::hasNegateOnOut(ExprId e) const {
  const auto &expr = exp(e);
  switch (expr.kind) {
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
    return expContainsTensor(expr.children.e0, outTensor);
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
    return expContainsTensor(expr.children.e1, outTensor) ||
           hasNegateOnOut(expr.children.e0);
  default: {
    switch (getExpArity(expr.kind)) {
    case ExpArity::kNullary:
      return false;
    case ExpArity::kUnary:
      return hasNegateOnOut(expr.children.e0);
    case ExpArity::kBinary:
      return hasNegateOnOut(expr.children.e0) ||
             hasNegateOnOut(expr.children.e1);
    }
  }
  }
  llvm_unreachable("unexpected kind");
}

bool Merger::isSingleCondition(TensorId t, ExprId e) const {
  assert(isValidTensorId(t));
  const auto &expr = exp(e);
  switch (expr.kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
    return expr.tensor == t;
  case TensorExp::Kind::kInvariant:
  case TensorExp::Kind::kLoopVar:
    return false;
  // Unary operations.
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
  case TensorExp::Kind::kCeilF:
  case TensorExp::Kind::kFloorF:
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kCIm:
  case TensorExp::Kind::kCRe:
  case TensorExp::Kind::kBitCast:
    return isSingleCondition(t, expr.children.e0);
  case TensorExp::Kind::kBinaryBranch:
  case TensorExp::Kind::kUnary:
  case TensorExp::Kind::kSelect:
    return false;
  // Binary operations.
  case TensorExp::Kind::kDivF: // note: x / c only
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
    assert(!maybeZero(expr.children.e1));
    return isSingleCondition(t, expr.children.e0);
  case TensorExp::Kind::kShrS: // note: x >> inv only
  case TensorExp::Kind::kShrU:
  case TensorExp::Kind::kShlI:
    assert(isInvariant(expr.children.e1));
    return isSingleCondition(t, expr.children.e0);
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
  case TensorExp::Kind::kAndI:
    if (isSingleCondition(t, expr.children.e0))
      return isSingleCondition(t, expr.children.e1) ||
             isInvariant(expr.children.e1);
    if (isSingleCondition(t, expr.children.e1))
      return isInvariant(expr.children.e0);
    return false;
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
    return isSingleCondition(t, expr.children.e0) &&
           isSingleCondition(t, expr.children.e1);
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
  case TensorExp::Kind::kOrI:
  case TensorExp::Kind::kXorI:
  case TensorExp::Kind::kBinary:
  case TensorExp::Kind::kReduce:
    return false;
  }
  llvm_unreachable("unexpected kind");
}

bool Merger::hasAnySparse(const BitVector &bits) const {
  for (TensorLoopId b : bits.set_bits()) {
    const auto dlt = getDimLevelType(b);
    if (isCompressedDLT(dlt) || isSingletonDLT(dlt))
      return true;
  }
  return hasSparseIdxReduction(bits);
}

bool Merger::hasSparseIdxReduction(const BitVector &bits) const {
  for (TensorLoopId b : bits.set_bits())
    if (isSparseLvlWithNonTrivialIdxExp(b))
      return true;
  return false;
}

#ifndef NDEBUG

//===----------------------------------------------------------------------===//
// Print methods (for debugging).
//===----------------------------------------------------------------------===//

static const char *kindToOpSymbol(TensorExp::Kind kind) {
  switch (kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
    return "tensor";
  case TensorExp::Kind::kInvariant:
    return "invariant";
  case TensorExp::Kind::kLoopVar:
    return "index";
  // Unary operations.
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
    return "abs";
  case TensorExp::Kind::kCeilF:
    return "ceil";
  case TensorExp::Kind::kFloorF:
    return "floor";
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
    return "sqrt";
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
    return "expm1";
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
    return "log1p";
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
    return "sin";
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
    return "tanh";
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
    return "-";
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kCIm:
    return "complex.im";
  case TensorExp::Kind::kCRe:
    return "complex.re";
  case TensorExp::Kind::kBitCast:
    return "cast";
  case TensorExp::Kind::kBinaryBranch:
    return "binary_branch";
  case TensorExp::Kind::kUnary:
    return "unary";
  case TensorExp::Kind::kSelect:
    return "select";
  // Binary operations.
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
    return "*";
  case TensorExp::Kind::kDivF:
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
    return "/";
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
    return "+";
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
    return "-";
  case TensorExp::Kind::kAndI:
    return "&";
  case TensorExp::Kind::kOrI:
    return "|";
  case TensorExp::Kind::kXorI:
    return "^";
  case TensorExp::Kind::kShrS:
    return "a>>";
  case TensorExp::Kind::kShrU:
    return ">>";
  case TensorExp::Kind::kShlI:
    return "<<";
  case TensorExp::Kind::kBinary:
    return "binary";
  case TensorExp::Kind::kReduce:
    return "reduce";
  }
  llvm_unreachable("unexpected kind for symbol");
}

void Merger::dumpExp(ExprId e) const {
  const auto &expr = exp(e);
  switch (expr.kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
    if (expr.tensor == syntheticTensor)
      llvm::dbgs() << "synthetic_";
    else if (expr.tensor == outTensor)
      llvm::dbgs() << "output_";
    llvm::dbgs() << "tensor_" << expr.tensor;
    break;
  case TensorExp::Kind::kInvariant:
    llvm::dbgs() << "invariant";
    break;
  case TensorExp::Kind::kLoopVar:
    llvm::dbgs() << "loopvar_" << expr.loop;
    break;
  // Unary operations.
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
  case TensorExp::Kind::kCeilF:
  case TensorExp::Kind::kFloorF:
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kCIm:
  case TensorExp::Kind::kCRe:
  case TensorExp::Kind::kBitCast:
  case TensorExp::Kind::kBinaryBranch:
  case TensorExp::Kind::kUnary:
  case TensorExp::Kind::kSelect:
    llvm::dbgs() << kindToOpSymbol(expr.kind) << " ";
    dumpExp(expr.children.e0);
    break;
  // Binary operations.
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
  case TensorExp::Kind::kDivF:
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
  case TensorExp::Kind::kAndI:
  case TensorExp::Kind::kOrI:
  case TensorExp::Kind::kXorI:
  case TensorExp::Kind::kShrS:
  case TensorExp::Kind::kShrU:
  case TensorExp::Kind::kShlI:
  case TensorExp::Kind::kBinary:
  case TensorExp::Kind::kReduce:
    llvm::dbgs() << "(";
    dumpExp(expr.children.e0);
    llvm::dbgs() << " " << kindToOpSymbol(expr.kind) << " ";
    dumpExp(expr.children.e1);
    llvm::dbgs() << ")";
  }
}

void Merger::dumpLat(LatPointId p) const {
  const auto &point = lat(p);
  llvm::dbgs() << "lat(";
  dumpBits(point.bits);
  llvm::dbgs() << " :";
  dumpBits(point.simple);
  llvm::dbgs() << " : ";
  dumpExp(point.exp);
  llvm::dbgs() << " )\n";
}

void Merger::dumpSet(LatSetId s) const {
  const auto &ss = set(s);
  llvm::dbgs() << "{ #" << ss.size() << "\n";
  for (const LatPointId p : ss) {
    llvm::dbgs() << "  ";
    dumpLat(p);
  }
  llvm::dbgs() << "}\n";
}

void Merger::dumpBits(const BitVector &bits) const {
  for (TensorLoopId b = 0, be = bits.size(); b < be; b++) {
    if (bits[b]) {
      const TensorId t = tensor(b);
      const LoopId i = loop(b);
      const auto dlt = lvlTypes[t][i];
      if (isLvlWithNonTrivialIdxExp(b))
        llvm::dbgs() << " DEP_" << t << "_" << i;
      else
        llvm::dbgs() << " i_" << t << "_" << i << "_" << toMLIRString(dlt);
    }
  }
}

#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Builder methods.
//===----------------------------------------------------------------------===//

LatSetId Merger::buildLattices(ExprId e, LoopId i) {
  // NOTE: The `expr` reference will be invalidated by recursive calls
  // (and any other method that may add new expressions); therefore, the
  // code below must make sure to copy fields of `expr` into local variables
  // before making any recursive calls.
  const auto &expr = exp(e);
  const TensorExp::Kind kind = expr.kind;
  switch (kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
  case TensorExp::Kind::kInvariant:
  case TensorExp::Kind::kLoopVar: {
    // Either the loop-var is really used in the tensor expression, or it is
    // set to the undefined loop-var in that level. An invariant expression,
    // a proper index value, and a truly dynamic sparse output tensor are set
    // to a synthetic tensor with undefined indices only to ensure the
    // iteration space is not skipped as a result of their contents.
    const LatSetId s = addSet();
    TensorId t = syntheticTensor;
    if (kind == TensorExp::Kind::kTensor) {
      t = expr.tensor;
      if (hasSparseOut && t == outTensor)
        t = syntheticTensor;
    }
    latSets[s].push_back(addLat(t, i, e));
    return s;
  }
  // Unary operations.
  case TensorExp::Kind::kAbsF:
  case TensorExp::Kind::kAbsC:
  case TensorExp::Kind::kAbsI:
  case TensorExp::Kind::kCeilF:
  case TensorExp::Kind::kFloorF:
  case TensorExp::Kind::kSqrtF:
  case TensorExp::Kind::kSqrtC:
  case TensorExp::Kind::kExpm1F:
  case TensorExp::Kind::kExpm1C:
  case TensorExp::Kind::kLog1pF:
  case TensorExp::Kind::kLog1pC:
  case TensorExp::Kind::kSinF:
  case TensorExp::Kind::kSinC:
  case TensorExp::Kind::kTanhF:
  case TensorExp::Kind::kTanhC:
  case TensorExp::Kind::kNegF:
  case TensorExp::Kind::kNegC:
  case TensorExp::Kind::kNegI:
  case TensorExp::Kind::kTruncF:
  case TensorExp::Kind::kExtF:
  case TensorExp::Kind::kCastFS:
  case TensorExp::Kind::kCastFU:
  case TensorExp::Kind::kCastSF:
  case TensorExp::Kind::kCastUF:
  case TensorExp::Kind::kCastS:
  case TensorExp::Kind::kCastU:
  case TensorExp::Kind::kCastIdx:
  case TensorExp::Kind::kTruncI:
  case TensorExp::Kind::kCIm:
  case TensorExp::Kind::kCRe:
  case TensorExp::Kind::kBitCast:
    // A zero preserving operation (viz. f(0) = 0, [Bik96,Ch5]) maps the
    // lattice set of the operand through the operator into a new set.
    //
    //  -y|!y | y |
    //  --+---+---+
    //    | 0 |-y |
    {
      const ExprId e0 = expr.children.e0;
      const Value v = expr.val;
      return mapSet(kind, buildLattices(e0, i), v);
    }
  case TensorExp::Kind::kBinaryBranch:
  case TensorExp::Kind::kSelect:
    // The left or right half of a binary operation which has already
    // been split into separate operations for each region.
    {
      const ExprId e0 = expr.children.e0;
      Operation *const op = expr.op;
      return mapSet(kind, buildLattices(e0, i), Value(), op);
    }
  case TensorExp::Kind::kUnary:
    // A custom unary operation.
    //
    //  op y|    !y    |     y      |
    //  ----+----------+------------+
    //      | absent() | present(y) |
    {
      const ExprId e0 = expr.children.e0;
      UnaryOp unop = cast<UnaryOp>(expr.op);
      const LatSetId child0 = buildLattices(e0, i);
      Region &absentRegion = unop.getAbsentRegion();

      if (absentRegion.empty()) {
        // Simple mapping over existing values.
        return mapSet(kind, child0, Value(), unop);
      } // Use a disjunction with `unop` on the left and the absent value as an
      // invariant on the right.
      Block &absentBlock = absentRegion.front();
      YieldOp absentYield = cast<YieldOp>(absentBlock.getTerminator());
      const Value absentVal = absentYield.getResult();
      const ExprId rhs = addInvariantExp(absentVal);
      return disjSet(kind, child0, buildLattices(rhs, i), unop);
    }
  // Binary operations.
  case TensorExp::Kind::kMulF:
  case TensorExp::Kind::kMulC:
  case TensorExp::Kind::kMulI:
  case TensorExp::Kind::kAndI:
    // A multiplicative operation only needs to be performed
    // for the conjunction of sparse iteration spaces.
    //
    //  x*y|!y | y |
    //  ---+---+---+
    //  !x | 0 | 0 |
    //   x | 0 |x*y|
    //
    // Note even here, 0*NaN=NaN and 0*Inf=NaN, but that is ignored.
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      return conjSet(kind, buildLattices(e0, i), buildLattices(e1, i));
    }
  case TensorExp::Kind::kDivF:
  case TensorExp::Kind::kDivC:
  case TensorExp::Kind::kDivS:
  case TensorExp::Kind::kDivU:
    // A division is tricky, since 0/0, 0/c, c/0 all have
    // specific outcomes for floating-point and integers.
    // Thus, we need to traverse the full iteration space.
    //
    //  x/y|!y | y |
    //  ---+---+---+
    //  !x |0/0|0/y|   FP: 0/0=NaN,c/0=Inf,0/c=0 with c true nonzero
    //   x |x/0|x/y|  INT: x/0=exception for any x
    //
    // TODO: for now we "fixed" this by only accepting x/c cases
    //       during expression building, so that the conjunction
    //       rules applies (viz. x/c = x*(1/c) as far as lattice
    //       construction is concerned).
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      assert(!maybeZero(e1));
      return conjSet(kind, buildLattices(e0, i), buildLattices(e1, i));
    }
  case TensorExp::Kind::kAddF:
  case TensorExp::Kind::kAddC:
  case TensorExp::Kind::kAddI:
  case TensorExp::Kind::kSubF:
  case TensorExp::Kind::kSubC:
  case TensorExp::Kind::kSubI:
  case TensorExp::Kind::kOrI:
  case TensorExp::Kind::kXorI:
    // An additive operation needs to be performed
    // for the disjunction of sparse iteration spaces.
    //
    //  x+y|!y | y |    x-y|!y | y |
    //  ---+---+---+    ---+---+---+
    //  !x | 0 | y |    !x | 0 |-y |
    //   x | x |x+y|     x | x |x-y|
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      return disjSet(kind, buildLattices(e0, i), buildLattices(e1, i));
    }
  case TensorExp::Kind::kShrS:
  case TensorExp::Kind::kShrU:
  case TensorExp::Kind::kShlI:
    // A shift operation by an invariant amount (viz. tensor expressions
    // can only occur at the left-hand-side of the operator) can be handled
    // with the conjuction rule.
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      assert(isInvariant(e1));
      return conjSet(kind, buildLattices(e0, i), buildLattices(e1, i));
    }
  case TensorExp::Kind::kBinary:
    // A custom binary operation.
    //
    //  x op y|   !y    |       y      |
    //  ------+---------+--------------+
    //    !x  |  empty  |   right(y)   |
    //     x  | left(x) | overlap(x,y) |
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      BinaryOp binop = cast<BinaryOp>(expr.op);
      const LatSetId child0 = buildLattices(e0, i);
      const LatSetId child1 = buildLattices(e1, i);
      Region &leftRegion = binop.getLeftRegion();
      Region &rightRegion = binop.getRightRegion();
      // Left Region.
      Operation *leftYield = nullptr;
      if (!leftRegion.empty()) {
        Block &leftBlock = leftRegion.front();
        leftYield = leftBlock.getTerminator();
      }
      // Right Region.
      Operation *rightYield = nullptr;
      if (!rightRegion.empty()) {
        Block &rightBlock = rightRegion.front();
        rightYield = rightBlock.getTerminator();
      }
      bool includeLeft = binop.getLeftIdentity() || !leftRegion.empty();
      bool includeRight = binop.getRightIdentity() || !rightRegion.empty();
      return combiSet(TensorExp::Kind::kBinary, child0, child1, binop,
                      includeLeft, TensorExp::Kind::kBinaryBranch, leftYield,
                      includeRight, TensorExp::Kind::kBinaryBranch, rightYield);
    }
  case TensorExp::Kind::kReduce:
    // A custom reduce operation.
    {
      const ExprId e0 = expr.children.e0;
      const ExprId e1 = expr.children.e1;
      Operation *const op = expr.op;
      return conjSet(kind, buildLattices(e0, i), buildLattices(e1, i), op);
    }
  }
  llvm_unreachable("unexpected expression kind");
}

std::optional<ExprId> Merger::buildTensorExpFromLinalg(linalg::GenericOp op) {
  // Build the linalg semantics backward from yield.
  Operation *yield = op.getRegion().front().getTerminator();
  assert(isa<linalg::YieldOp>(yield));
  return buildTensorExp(op, yield->getOperand(0));
}

/// Only returns false if we are certain this is a nonzero.
bool Merger::maybeZero(ExprId e) const {
  const auto &expr = exp(e);
  if (expr.kind == TensorExp::Kind::kInvariant) {
    if (auto c = expr.val.getDefiningOp<complex::ConstantOp>()) {
      ArrayAttr arrayAttr = c.getValue();
      return arrayAttr[0].cast<FloatAttr>().getValue().isZero() &&
             arrayAttr[1].cast<FloatAttr>().getValue().isZero();
    }
    if (auto c = expr.val.getDefiningOp<arith::ConstantIntOp>())
      return c.value() == 0;
    if (auto c = expr.val.getDefiningOp<arith::ConstantFloatOp>())
      return c.value().isZero();
  }
  return true;
}

Type Merger::inferType(ExprId e, Value src) const {
  // Obtain the destination type from the cast node.
  Type dtp = exp(e).val.getType();
  // Inspect source type. For vector types, apply the same
  // vectorization to the destination type.
  if (auto vtp = src.getType().dyn_cast<VectorType>())
    return VectorType::get(vtp.getNumElements(), dtp, vtp.getNumScalableDims());
  return dtp;
}

/// Ensures that sparse compiler can generate code for expression.
static bool isAdmissibleBranchExp(Operation *op, Block *block, Value v) {
  // Arguments are always admissible.
  if (v.isa<BlockArgument>())
    return true;
  // Accept index anywhere.
  Operation *def = v.getDefiningOp();
  if (isa<linalg::IndexOp>(def))
    return true;
  // Operation defined outside branch.
  if (def->getBlock() != block)
    return def->getBlock() != op->getBlock(); // invariant?
  // Operation defined within branch. Anything is accepted,
  // as long as all subexpressions are admissible.
  for (unsigned i = 0, n = def->getNumOperands(); i < n; i++)
    if (!isAdmissibleBranchExp(op, block, def->getOperand(i)))
      return false;
  return true;
}

/// Ensures that sparse compiler can generate code for branch.
static bool isAdmissibleBranch(Operation *op, Region &region) {
  if (region.empty())
    return true;
  // Build the semi-ring branch semantics backward from yield.
  Operation *yield = region.front().getTerminator();
  assert(isa<YieldOp>(yield));
  return isAdmissibleBranchExp(op, &region.front(), yield->getOperand(0));
}

std::optional<ExprId> Merger::buildTensorExp(linalg::GenericOp op, Value v) {
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    const TensorId tid = makeTensorId(arg.getArgNumber());
    // Any argument of the generic op that is not marked as a scalar
    // argument is considered a tensor, indexed by the implicit loop
    // bounds. This includes rank-0 tensor arguments.
    if (arg.getOwner()->getParentOp() == op) {
      OpOperand &t = op->getOpOperand(tid);
      if (!op.isScalar(&t))
        return addTensorExp(tid);
      v = t.get(); // get scalar value
    }
    // Any other argument (marked as scalar argument for the generic op
    // or belonging to an enveloping op) is considered invariant.
    return addInvariantExp(v);
  }
  // Something defined outside is invariant.
  Operation *def = v.getDefiningOp();
  if (def->getBlock() != &op.getRegion().front())
    return addInvariantExp(v);
  // Construct index operations.
  if (def->getNumOperands() == 0) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return addLoopVarExp(makeLoopId(indexOp.getDim()));
  }
  // Construct unary operations if subexpression can be built.
  if (def->getNumOperands() == 1) {
    const auto x = buildTensorExp(op, def->getOperand(0));
    if (x.has_value()) {
      const ExprId e = *x;
      if (isa<math::AbsFOp>(def))
        return addExp(TensorExp::Kind::kAbsF, e);
      if (isa<complex::AbsOp>(def))
        return addExp(TensorExp::Kind::kAbsC, e);
      if (isa<math::AbsIOp>(def))
        return addExp(TensorExp::Kind::kAbsI, e);
      if (isa<math::CeilOp>(def))
        return addExp(TensorExp::Kind::kCeilF, e);
      if (isa<math::FloorOp>(def))
        return addExp(TensorExp::Kind::kFloorF, e);
      if (isa<math::SqrtOp>(def))
        return addExp(TensorExp::Kind::kSqrtF, e);
      if (isa<complex::SqrtOp>(def))
        return addExp(TensorExp::Kind::kSqrtC, e);
      if (isa<math::ExpM1Op>(def))
        return addExp(TensorExp::Kind::kExpm1F, e);
      if (isa<complex::Expm1Op>(def))
        return addExp(TensorExp::Kind::kExpm1C, e);
      if (isa<math::Log1pOp>(def))
        return addExp(TensorExp::Kind::kLog1pF, e);
      if (isa<complex::Log1pOp>(def))
        return addExp(TensorExp::Kind::kLog1pC, e);
      if (isa<math::SinOp>(def))
        return addExp(TensorExp::Kind::kSinF, e);
      if (isa<complex::SinOp>(def))
        return addExp(TensorExp::Kind::kSinC, e);
      if (isa<math::TanhOp>(def))
        return addExp(TensorExp::Kind::kTanhF, e);
      if (isa<complex::TanhOp>(def))
        return addExp(TensorExp::Kind::kTanhC, e);
      if (isa<arith::NegFOp>(def))
        return addExp(TensorExp::Kind::kNegF, e); // no negi in std
      if (isa<complex::NegOp>(def))
        return addExp(TensorExp::Kind::kNegC, e);
      if (isa<arith::TruncFOp>(def))
        return addExp(TensorExp::Kind::kTruncF, e, v);
      if (isa<arith::ExtFOp>(def))
        return addExp(TensorExp::Kind::kExtF, e, v);
      if (isa<arith::FPToSIOp>(def))
        return addExp(TensorExp::Kind::kCastFS, e, v);
      if (isa<arith::FPToUIOp>(def))
        return addExp(TensorExp::Kind::kCastFU, e, v);
      if (isa<arith::SIToFPOp>(def))
        return addExp(TensorExp::Kind::kCastSF, e, v);
      if (isa<arith::UIToFPOp>(def))
        return addExp(TensorExp::Kind::kCastUF, e, v);
      if (isa<arith::ExtSIOp>(def))
        return addExp(TensorExp::Kind::kCastS, e, v);
      if (isa<arith::ExtUIOp>(def))
        return addExp(TensorExp::Kind::kCastU, e, v);
      if (isa<arith::IndexCastOp>(def))
        return addExp(TensorExp::Kind::kCastIdx, e, v);
      if (isa<arith::TruncIOp>(def))
        return addExp(TensorExp::Kind::kTruncI, e, v);
      if (isa<complex::ImOp>(def))
        return addExp(TensorExp::Kind::kCIm, e);
      if (isa<complex::ReOp>(def))
        return addExp(TensorExp::Kind::kCRe, e);
      if (isa<arith::BitcastOp>(def))
        return addExp(TensorExp::Kind::kBitCast, e, v);
      if (auto unop = dyn_cast<sparse_tensor::UnaryOp>(def)) {
        if (isAdmissibleBranch(unop, unop.getPresentRegion()) &&
            isAdmissibleBranch(unop, unop.getAbsentRegion()))
          return addExp(TensorExp::Kind::kUnary, e, Value(), def);
      }
      if (auto selop = dyn_cast<sparse_tensor::SelectOp>(def)) {
        if (isAdmissibleBranch(selop, selop.getRegion()))
          return addExp(TensorExp::Kind::kSelect, e, Value(), def);
      }
    }
  }
  // Construct binary operations if subexpressions can be built.
  // See buildLattices() for an explanation of rejecting certain
  // division and shift operations.
  if (def->getNumOperands() == 2) {
    const auto x = buildTensorExp(op, def->getOperand(0));
    const auto y = buildTensorExp(op, def->getOperand(1));
    if (x.has_value() && y.has_value()) {
      const ExprId e0 = *x;
      const ExprId e1 = *y;
      if (isa<arith::MulFOp>(def))
        return addExp(TensorExp::Kind::kMulF, e0, e1);
      if (isa<complex::MulOp>(def))
        return addExp(TensorExp::Kind::kMulC, e0, e1);
      if (isa<arith::MulIOp>(def))
        return addExp(TensorExp::Kind::kMulI, e0, e1);
      if (isa<arith::DivFOp>(def) && !maybeZero(e1))
        return addExp(TensorExp::Kind::kDivF, e0, e1);
      if (isa<complex::DivOp>(def) && !maybeZero(e1))
        return addExp(TensorExp::Kind::kDivC, e0, e1);
      if (isa<arith::DivSIOp>(def) && !maybeZero(e1))
        return addExp(TensorExp::Kind::kDivS, e0, e1);
      if (isa<arith::DivUIOp>(def) && !maybeZero(e1))
        return addExp(TensorExp::Kind::kDivU, e0, e1);
      if (isa<arith::AddFOp>(def))
        return addExp(TensorExp::Kind::kAddF, e0, e1);
      if (isa<complex::AddOp>(def))
        return addExp(TensorExp::Kind::kAddC, e0, e1);
      if (isa<arith::AddIOp>(def))
        return addExp(TensorExp::Kind::kAddI, e0, e1);
      if (isa<arith::SubFOp>(def))
        return addExp(TensorExp::Kind::kSubF, e0, e1);
      if (isa<complex::SubOp>(def))
        return addExp(TensorExp::Kind::kSubC, e0, e1);
      if (isa<arith::SubIOp>(def))
        return addExp(TensorExp::Kind::kSubI, e0, e1);
      if (isa<arith::AndIOp>(def))
        return addExp(TensorExp::Kind::kAndI, e0, e1);
      if (isa<arith::OrIOp>(def))
        return addExp(TensorExp::Kind::kOrI, e0, e1);
      if (isa<arith::XOrIOp>(def))
        return addExp(TensorExp::Kind::kXorI, e0, e1);
      if (isa<arith::ShRSIOp>(def) && isInvariant(e1))
        return addExp(TensorExp::Kind::kShrS, e0, e1);
      if (isa<arith::ShRUIOp>(def) && isInvariant(e1))
        return addExp(TensorExp::Kind::kShrU, e0, e1);
      if (isa<arith::ShLIOp>(def) && isInvariant(e1))
        return addExp(TensorExp::Kind::kShlI, e0, e1);
      if (auto binop = dyn_cast<sparse_tensor::BinaryOp>(def)) {
        if (isAdmissibleBranch(binop, binop.getOverlapRegion()) &&
            (binop.getLeftIdentity() ||
             isAdmissibleBranch(binop, binop.getLeftRegion())) &&
            (binop.getRightIdentity() ||
             isAdmissibleBranch(binop, binop.getRightRegion())))
          return addExp(TensorExp::Kind::kBinary, e0, e1, def);
      }
    }
  }
  // Construct ternary operations if subexpressions can be built.
  if (def->getNumOperands() == 3) {
    const auto x = buildTensorExp(op, def->getOperand(0));
    const auto y = buildTensorExp(op, def->getOperand(1));
    const auto z = buildTensorExp(op, def->getOperand(2));
    if (x.has_value() && y.has_value() && z.has_value()) {
      const ExprId e0 = *x;
      const ExprId e1 = *y;
      if (auto redop = dyn_cast<sparse_tensor::ReduceOp>(def)) {
        if (isAdmissibleBranch(redop, redop.getRegion()))
          return addExp(TensorExp::Kind::kReduce, e0, e1, def);
      }
    }
  }
  // Cannot build.
  return std::nullopt;
}

static Value insertYieldOp(RewriterBase &rewriter, Location loc, Region &region,
                           ValueRange vals) {
  // Make a clone of overlap region.
  Region tmpRegion;
  IRMapping mapper;
  region.cloneInto(&tmpRegion, tmpRegion.begin(), mapper);
  Block &clonedBlock = tmpRegion.front();
  YieldOp clonedYield = cast<YieldOp>(clonedBlock.getTerminator());
  // Merge cloned block and return yield value.
  Operation *placeholder = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.inlineBlockBefore(&tmpRegion.front(), placeholder, vals);
  Value val = clonedYield.getResult();
  rewriter.eraseOp(clonedYield);
  rewriter.eraseOp(placeholder);
  return val;
}

static Value buildUnaryPresent(RewriterBase &rewriter, Location loc,
                               Operation *op, Value v0) {
  if (!v0)
    // Empty input value must be propagated.
    return Value();
  UnaryOp unop = cast<UnaryOp>(op);
  Region &presentRegion = unop.getPresentRegion();
  if (presentRegion.empty())
    // Uninitialized Value() will be interpreted as missing data in the
    // output.
    return Value();
  return insertYieldOp(rewriter, loc, presentRegion, {v0});
}

static Value buildBinaryOverlap(RewriterBase &rewriter, Location loc,
                                Operation *op, Value v0, Value v1) {
  if (!v0 || !v1)
    // Empty input values must be propagated.
    return Value();
  BinaryOp binop = cast<BinaryOp>(op);
  Region &overlapRegion = binop.getOverlapRegion();
  if (overlapRegion.empty())
    // Uninitialized Value() will be interpreted as missing data in the
    // output.
    return Value();
  return insertYieldOp(rewriter, loc, overlapRegion, {v0, v1});
}

Value Merger::buildExp(RewriterBase &rewriter, Location loc, ExprId e, Value v0,
                       Value v1) const {
  const auto &expr = exp(e);
  switch (expr.kind) {
  // Leaf.
  case TensorExp::Kind::kTensor:
  case TensorExp::Kind::kInvariant:
  case TensorExp::Kind::kLoopVar:
    llvm_unreachable("unexpected non-op");
  // Unary operations.
  case TensorExp::Kind::kAbsF:
    return rewriter.create<math::AbsFOp>(loc, v0);
  case TensorExp::Kind::kAbsC: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::AbsOp>(loc, eltType, v0);
  }
  case TensorExp::Kind::kAbsI:
    return rewriter.create<math::AbsIOp>(loc, v0);
  case TensorExp::Kind::kCeilF:
    return rewriter.create<math::CeilOp>(loc, v0);
  case TensorExp::Kind::kFloorF:
    return rewriter.create<math::FloorOp>(loc, v0);
  case TensorExp::Kind::kSqrtF:
    return rewriter.create<math::SqrtOp>(loc, v0);
  case TensorExp::Kind::kSqrtC:
    return rewriter.create<complex::SqrtOp>(loc, v0);
  case TensorExp::Kind::kExpm1F:
    return rewriter.create<math::ExpM1Op>(loc, v0);
  case TensorExp::Kind::kExpm1C:
    return rewriter.create<complex::Expm1Op>(loc, v0);
  case TensorExp::Kind::kLog1pF:
    return rewriter.create<math::Log1pOp>(loc, v0);
  case TensorExp::Kind::kLog1pC:
    return rewriter.create<complex::Log1pOp>(loc, v0);
  case TensorExp::Kind::kSinF:
    return rewriter.create<math::SinOp>(loc, v0);
  case TensorExp::Kind::kSinC:
    return rewriter.create<complex::SinOp>(loc, v0);
  case TensorExp::Kind::kTanhF:
    return rewriter.create<math::TanhOp>(loc, v0);
  case TensorExp::Kind::kTanhC:
    return rewriter.create<complex::TanhOp>(loc, v0);
  case TensorExp::Kind::kNegF:
    return rewriter.create<arith::NegFOp>(loc, v0);
  case TensorExp::Kind::kNegC:
    return rewriter.create<complex::NegOp>(loc, v0);
  case TensorExp::Kind::kNegI: // no negi in std
    return rewriter.create<arith::SubIOp>(
        loc,
        rewriter.create<arith::ConstantOp>(loc, v0.getType(),
                                           rewriter.getZeroAttr(v0.getType())),
        v0);
  case TensorExp::Kind::kTruncF:
    return rewriter.create<arith::TruncFOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kExtF:
    return rewriter.create<arith::ExtFOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastFS:
    return rewriter.create<arith::FPToSIOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastFU:
    return rewriter.create<arith::FPToUIOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastSF:
    return rewriter.create<arith::SIToFPOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastUF:
    return rewriter.create<arith::UIToFPOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastS:
    return rewriter.create<arith::ExtSIOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastU:
    return rewriter.create<arith::ExtUIOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCastIdx:
    return rewriter.create<arith::IndexCastOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kTruncI:
    return rewriter.create<arith::TruncIOp>(loc, inferType(e, v0), v0);
  case TensorExp::Kind::kCIm: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::ImOp>(loc, eltType, v0);
  }
  case TensorExp::Kind::kCRe: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::ReOp>(loc, eltType, v0);
  }
  case TensorExp::Kind::kBitCast:
    return rewriter.create<arith::BitcastOp>(loc, inferType(e, v0), v0);
  // Binary operations.
  case TensorExp::Kind::kMulF:
    return rewriter.create<arith::MulFOp>(loc, v0, v1);
  case TensorExp::Kind::kMulC:
    return rewriter.create<complex::MulOp>(loc, v0, v1);
  case TensorExp::Kind::kMulI:
    return rewriter.create<arith::MulIOp>(loc, v0, v1);
  case TensorExp::Kind::kDivF:
    return rewriter.create<arith::DivFOp>(loc, v0, v1);
  case TensorExp::Kind::kDivC:
    return rewriter.create<complex::DivOp>(loc, v0, v1);
  case TensorExp::Kind::kDivS:
    return rewriter.create<arith::DivSIOp>(loc, v0, v1);
  case TensorExp::Kind::kDivU:
    return rewriter.create<arith::DivUIOp>(loc, v0, v1);
  case TensorExp::Kind::kAddF:
    return rewriter.create<arith::AddFOp>(loc, v0, v1);
  case TensorExp::Kind::kAddC:
    return rewriter.create<complex::AddOp>(loc, v0, v1);
  case TensorExp::Kind::kAddI:
    return rewriter.create<arith::AddIOp>(loc, v0, v1);
  case TensorExp::Kind::kSubF:
    return rewriter.create<arith::SubFOp>(loc, v0, v1);
  case TensorExp::Kind::kSubC:
    return rewriter.create<complex::SubOp>(loc, v0, v1);
  case TensorExp::Kind::kSubI:
    return rewriter.create<arith::SubIOp>(loc, v0, v1);
  case TensorExp::Kind::kAndI:
    return rewriter.create<arith::AndIOp>(loc, v0, v1);
  case TensorExp::Kind::kOrI:
    return rewriter.create<arith::OrIOp>(loc, v0, v1);
  case TensorExp::Kind::kXorI:
    return rewriter.create<arith::XOrIOp>(loc, v0, v1);
  case TensorExp::Kind::kShrS:
    return rewriter.create<arith::ShRSIOp>(loc, v0, v1);
  case TensorExp::Kind::kShrU:
    return rewriter.create<arith::ShRUIOp>(loc, v0, v1);
  case TensorExp::Kind::kShlI:
    return rewriter.create<arith::ShLIOp>(loc, v0, v1);
  case TensorExp::Kind::kBinaryBranch: // semi-ring ops with custom logic.
    return insertYieldOp(rewriter, loc, *expr.op->getBlock()->getParent(),
                         {v0});
  case TensorExp::Kind::kUnary:
    return buildUnaryPresent(rewriter, loc, expr.op, v0);
  case TensorExp::Kind::kSelect:
    return insertYieldOp(rewriter, loc, cast<SelectOp>(expr.op).getRegion(),
                         {v0});
  case TensorExp::Kind::kBinary:
    return buildBinaryOverlap(rewriter, loc, expr.op, v0, v1);
  case TensorExp::Kind::kReduce: {
    ReduceOp redOp = cast<ReduceOp>(expr.op);
    return insertYieldOp(rewriter, loc, redOp.getRegion(), {v0, v1});
  }
  }
  llvm_unreachable("unexpected expression kind in build");
}

} // namespace sparse_tensor
} // namespace mlir
