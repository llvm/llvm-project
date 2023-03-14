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

static ExpArity getExpArity(Kind k) {
  switch (k) {
  // Leaf.
  case kTensor:
  case kInvariant:
  case kLoopVar:
    return ExpArity::kNullary;
  case kAbsF:
  case kAbsC:
  case kAbsI:
  case kCeilF:
  case kFloorF:
  case kSqrtF:
  case kSqrtC:
  case kExpm1F:
  case kExpm1C:
  case kLog1pF:
  case kLog1pC:
  case kSinF:
  case kSinC:
  case kTanhF:
  case kTanhC:
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kCIm:
  case kCRe:
  case kBitCast:
  case kBinaryBranch:
  case kUnary:
  case kSelect:
  case kNegF:
  case kNegC:
  case kNegI:
    return ExpArity::kUnary;
  // Binary operations.
  case kDivF:
  case kDivC:
  case kDivS:
  case kDivU:
  case kShrS:
  case kShrU:
  case kShlI:
  case kMulF:
  case kMulC:
  case kMulI:
  case kAndI:
  case kAddF:
  case kAddC:
  case kAddI:
  case kOrI:
  case kXorI:
  case kBinary:
  case kReduce:
  case kSubF:
  case kSubC:
  case kSubI:
    return ExpArity::kBinary;
  }
  llvm_unreachable("unexpected kind");
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

TensorExp::TensorExp(Kind k, unsigned x, ExprId y, Value v, Operation *o)
    : kind(k), val(v), op(o) {
  switch (kind) {
  // Leaf.
  case kTensor:
    assert(x != kInvalidId && y == kInvalidId && !v && !o);
    tensor = x;
    break;
  case kInvariant:
    assert(x == kInvalidId && y == kInvalidId && v && !o);
    break;
  case kLoopVar:
    assert(x != kInvalidId && y == kInvalidId && !v && !o);
    loop = x;
    break;
  // Unary operations.
  case kAbsF:
  case kAbsC:
  case kAbsI:
  case kCeilF:
  case kFloorF:
  case kSqrtF:
  case kSqrtC:
  case kExpm1F:
  case kExpm1C:
  case kLog1pF:
  case kLog1pC:
  case kSinF:
  case kSinC:
  case kTanhF:
  case kTanhC:
  case kNegF:
  case kNegC:
  case kNegI:
  case kCIm:
  case kCRe:
    assert(x != kInvalidId && y == kInvalidId && !v && !o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kBitCast:
    assert(x != kInvalidId && y == kInvalidId && v && !o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kBinaryBranch:
  case kSelect:
    assert(x != kInvalidId && y == kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kUnary:
    // No assertion on y can be made, as the branching paths involve both
    // a unary (`mapSet`) and binary (`disjSet`) pathway.
    assert(x != kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    break;
  // Binary operations.
  case kMulF:
  case kMulC:
  case kMulI:
  case kDivF:
  case kDivC:
  case kDivS:
  case kDivU:
  case kAddF:
  case kAddC:
  case kAddI:
  case kSubF:
  case kSubC:
  case kSubI:
  case kAndI:
  case kOrI:
  case kXorI:
  case kShrS:
  case kShrU:
  case kShlI:
    assert(x != kInvalidId && y != kInvalidId && !v && !o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kBinary:
  case kReduce:
    assert(x != kInvalidId && y != kInvalidId && !v && o);
    children.e0 = x;
    children.e1 = y;
    break;
  }
}

LatPoint::LatPoint(const BitVector &bits, ExprId e) : bits(bits), exp(e) {}

LatPoint::LatPoint(unsigned numTensors, unsigned numLoops, TensorId t, LoopId i,
                   ExprId e)
    : bits(numLoops * numTensors, false), exp(e) {
  assert(t < numTensors && i < numLoops);
  const TensorLoopId b = numTensors * i + t;
  bits.set(b);
}

Merger::Merger(unsigned numInputOutputTensors, unsigned numNativeLoops,
               unsigned numFilterLoops)
    : outTensor(numInputOutputTensors - 1),
      syntheticTensor(numInputOutputTensors),
      numTensors(numInputOutputTensors + 1), numNativeLoops(numNativeLoops),
      numLoops(numNativeLoops + numFilterLoops), hasSparseOut(false),
      lvlTypes(numTensors,
               std::vector<DimLevelType>(numLoops, DimLevelType::Undef)),
      loopToLvl(numTensors,
                std::vector<std::optional<Level>>(numLoops, std::nullopt)),
      lvlToLoop(numTensors,
                std::vector<std::optional<LoopId>>(numLoops, std::nullopt)) {}

//===----------------------------------------------------------------------===//
// Lattice methods.
//===----------------------------------------------------------------------===//

ExprId Merger::addExp(Kind k, unsigned x, ExprId y, Value v, Operation *op) {
  const ExprId e = tensorExps.size();
  assert((k != kTensor || x < numTensors) && (k != kLoopVar || x < numLoops));
  tensorExps.emplace_back(k, x, y, v, op);
  return e;
}

LatPointId Merger::addLat(TensorId t, LoopId i, ExprId e) {
  assert(t < numTensors && i < numLoops);
  const LatPointId p = latPoints.size();
  latPoints.emplace_back(numTensors, numLoops, t, i, e);
  return p;
}

LatSetId Merger::addSet() {
  const LatSetId s = latSets.size();
  latSets.emplace_back();
  return s;
}

LatPointId Merger::conjLat(Kind kind, LatPointId p0, LatPointId p1,
                           Operation *op) {
  const LatPointId p = latPoints.size();
  BitVector bits(latPoints[p0].bits);
  bits |= latPoints[p1].bits;
  const ExprId e =
      addExp(kind, latPoints[p0].exp, latPoints[p1].exp, Value(), op);
  latPoints.emplace_back(bits, e);
  return p;
}

LatSetId Merger::conjSet(Kind kind, LatSetId s0, LatSetId s1, Operation *op) {
  const LatSetId s = addSet();
  for (const LatPointId p0 : latSets[s0])
    for (const LatPointId p1 : latSets[s1])
      latSets[s].push_back(conjLat(kind, p0, p1, op));
  return s;
}

LatSetId Merger::disjSet(Kind kind, LatSetId s0, LatSetId s1, Operation *op) {
  const LatSetId s = conjSet(kind, s0, s1, op);
  // Followed by all in s0.
  for (const LatPointId p : latSets[s0])
    latSets[s].push_back(p);
  // Map binary 0-y to unary -y.
  // TODO: move this if-else logic into buildLattices
  if (kind == kSubF)
    s1 = mapSet(kNegF, s1);
  else if (kind == kSubC)
    s1 = mapSet(kNegC, s1);
  else if (kind == kSubI)
    s1 = mapSet(kNegI, s1);
  // Followed by all in s1.
  for (const LatPointId p : latSets[s1])
    latSets[s].push_back(p);
  return s;
}

LatSetId Merger::combiSet(Kind kind, LatSetId s0, LatSetId s1, Operation *orig,
                          bool includeLeft, Kind ltrans, Operation *opleft,
                          bool includeRight, Kind rtrans, Operation *opright) {
  const LatSetId s = conjSet(kind, s0, s1, orig);
  // Left Region.
  if (includeLeft) {
    if (opleft)
      s0 = mapSet(ltrans, s0, Value(), opleft);
    for (const LatPointId p : latSets[s0])
      latSets[s].push_back(p);
  }
  // Right Region.
  if (includeRight) {
    if (opright)
      s1 = mapSet(rtrans, s1, Value(), opright);
    for (const LatPointId p : latSets[s1])
      latSets[s].push_back(p);
  }
  return s;
}

LatSetId Merger::mapSet(Kind kind, LatSetId s0, Value v, Operation *op) {
  assert(kAbsF <= kind && kind <= kSelect);
  const LatSetId s = addSet();
  for (const LatPointId p : latSets[s0]) {
    const ExprId e = addExp(kind, latPoints[p].exp, v, op);
    latPoints.emplace_back(latPoints[p].bits, e);
    latSets[s].push_back(latPoints.size() - 1);
  }
  return s;
}

LatSetId Merger::optimizeSet(LatSetId s0) {
  const LatSetId s = addSet();
  assert(!latSets[s0].empty());
  const LatPointId p0 = latSets[s0][0];
  for (const LatPointId p1 : latSets[s0]) {
    bool add = true;
    if (p0 != p1) {
      // Check whether this is a straightforward copy.
      const ExprId e = latPoints[p1].exp;
      if (expIsTensor(e, outTensor))
        continue;
      // Check whether this conjunction is already covered.
      for (const LatPointId p2 : latSets[s]) {
        assert(!latGT(p1, p2)); // Lj => Li would be bad
        if (onlyDenseDiff(p2, p1)) {
          add = false;
          break;
        }
      }
      assert(!add || latGT(p0, p1));
    }
    if (add)
      latSets[s].push_back(p1);
  }
  for (const LatPointId p : latSets[s])
    latPoints[p].simple = simplifyCond(s, p);
  return s;
}

BitVector Merger::simplifyCond(LatSetId s0, LatPointId p0) {
  // First determine if this lattice point is a *singleton*, i.e.,
  // the last point in a lattice, no other is less than this one.
  bool isSingleton = true;
  for (const LatPointId p1 : latSets[s0]) {
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
    for (TensorLoopId b = 0; b < be; b++) {
      if (simple[b] && isDenseDLT(getDimLevelType(b))) {
        offset = be - b - 1; // relative to the end
        break;
      }
    }

  // Now apply the two basic rules. We also iterate the bits reversely to always
  // keep the rightmost bit (which could possibly be a synthetic tensor).
  for (TensorLoopId b = be - 1 - offset, i = 0; i < be;
       b = b == 0 ? be - 1 : b - 1, i++) {
    if (simple[b]) {
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
  const BitVector &bitsi = latPoints[i].bits;
  const BitVector &bitsj = latPoints[j].bits;
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
  if (tensorExps[e].kind == kTensor)
    return tensorExps[e].tensor == t;

  switch (getExpArity(tensorExps[e].kind)) {
  case ExpArity::kNullary:
    return false;
  case ExpArity::kUnary: {
    const ExprId e0 = tensorExps[e].children.e0;
    if (expIsTensor(e0, t))
      return true;
    return expContainsTensor(e0, t);
  }
  case ExpArity::kBinary: {
    const ExprId e0 = tensorExps[e].children.e0;
    const ExprId e1 = tensorExps[e].children.e1;
    if (expIsTensor(e0, t) || expIsTensor(e1, t))
      return true;
    return expContainsTensor(e0, t) || expContainsTensor(e1, t);
  }
  }
  llvm_unreachable("unexpected arity");
}

bool Merger::hasNegateOnOut(ExprId e) const {
  switch (tensorExps[e].kind) {
  case kNegF:
  case kNegC:
  case kNegI:
    return expContainsTensor(tensorExps[e].children.e0, outTensor);
  case kSubF:
  case kSubC:
  case kSubI:
    return expContainsTensor(tensorExps[e].children.e1, outTensor) ||
           hasNegateOnOut(tensorExps[e].children.e0);
  default: {
    switch (getExpArity(tensorExps[e].kind)) {
    case ExpArity::kNullary:
      return false;
    case ExpArity::kUnary:
      return hasNegateOnOut(tensorExps[e].children.e0);
    case ExpArity::kBinary:
      return hasNegateOnOut(tensorExps[e].children.e0) ||
             hasNegateOnOut(tensorExps[e].children.e1);
    }
  }
  }
  llvm_unreachable("unexpected kind");
}

bool Merger::isSingleCondition(TensorId t, ExprId e) const {
  assert(t < numTensors && e < tensorExps.size());
  switch (tensorExps[e].kind) {
  // Leaf.
  case kTensor:
    return tensorExps[e].tensor == t;
  case kInvariant:
  case kLoopVar:
    return false;
  // Unary operations.
  case kAbsF:
  case kAbsC:
  case kAbsI:
  case kCeilF:
  case kFloorF:
  case kSqrtF:
  case kSqrtC:
  case kExpm1F:
  case kExpm1C:
  case kLog1pF:
  case kLog1pC:
  case kSinF:
  case kSinC:
  case kTanhF:
  case kTanhC:
  case kNegF:
  case kNegC:
  case kNegI:
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kCIm:
  case kCRe:
  case kBitCast:
    return isSingleCondition(t, tensorExps[e].children.e0);
  case kBinaryBranch:
  case kUnary:
  case kSelect:
    return false;
  // Binary operations.
  case kDivF: // note: x / c only
  case kDivC:
  case kDivS:
  case kDivU:
    assert(!maybeZero(tensorExps[e].children.e1));
    return isSingleCondition(t, tensorExps[e].children.e0);
  case kShrS: // note: x >> inv only
  case kShrU:
  case kShlI:
    assert(isInvariant(tensorExps[e].children.e1));
    return isSingleCondition(t, tensorExps[e].children.e0);
  case kMulF:
  case kMulC:
  case kMulI:
  case kAndI:
    if (isSingleCondition(t, tensorExps[e].children.e0))
      return isSingleCondition(t, tensorExps[e].children.e1) ||
             isInvariant(tensorExps[e].children.e1);
    if (isSingleCondition(t, tensorExps[e].children.e1))
      return isInvariant(tensorExps[e].children.e0);
    return false;
  case kAddF:
  case kAddC:
  case kAddI:
    return isSingleCondition(t, tensorExps[e].children.e0) &&
           isSingleCondition(t, tensorExps[e].children.e1);
  case kSubF:
  case kSubC:
  case kSubI:
  case kOrI:
  case kXorI:
  case kBinary:
  case kReduce:
    return false;
  }
  llvm_unreachable("unexpected kind");
}

bool Merger::hasAnySparse(const BitVector &bits) const {
  for (TensorLoopId b = 0, be = bits.size(); b < be; b++)
    if (bits[b]) {
      const auto dlt = getDimLevelType(b);
      if (isCompressedDLT(dlt) || isSingletonDLT(dlt))
        return true;
    }
  return false;
}

#ifndef NDEBUG

//===----------------------------------------------------------------------===//
// Print methods (for debugging).
//===----------------------------------------------------------------------===//

static const char *kindToOpSymbol(Kind kind) {
  switch (kind) {
  // Leaf.
  case kTensor:
    return "tensor";
  case kInvariant:
    return "invariant";
  case kLoopVar:
    return "index";
  // Unary operations.
  case kAbsF:
  case kAbsC:
  case kAbsI:
    return "abs";
  case kCeilF:
    return "ceil";
  case kFloorF:
    return "floor";
  case kSqrtF:
  case kSqrtC:
    return "sqrt";
  case kExpm1F:
  case kExpm1C:
    return "expm1";
  case kLog1pF:
  case kLog1pC:
    return "log1p";
  case kSinF:
  case kSinC:
    return "sin";
  case kTanhF:
  case kTanhC:
    return "tanh";
  case kNegF:
  case kNegC:
  case kNegI:
    return "-";
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kCIm:
    return "complex.im";
  case kCRe:
    return "complex.re";
  case kBitCast:
    return "cast";
  case kBinaryBranch:
    return "binary_branch";
  case kUnary:
    return "unary";
  case kSelect:
    return "select";
  // Binary operations.
  case kMulF:
  case kMulC:
  case kMulI:
    return "*";
  case kDivF:
  case kDivC:
  case kDivS:
  case kDivU:
    return "/";
  case kAddF:
  case kAddC:
  case kAddI:
    return "+";
  case kSubF:
  case kSubC:
  case kSubI:
    return "-";
  case kAndI:
    return "&";
  case kOrI:
    return "|";
  case kXorI:
    return "^";
  case kShrS:
    return "a>>";
  case kShrU:
    return ">>";
  case kShlI:
    return "<<";
  case kBinary:
    return "binary";
  case kReduce:
    return "reduce";
  }
  llvm_unreachable("unexpected kind for symbol");
}

void Merger::dumpExp(ExprId e) const {
  switch (tensorExps[e].kind) {
  // Leaf.
  case kTensor:
    if (tensorExps[e].tensor == syntheticTensor)
      llvm::dbgs() << "synthetic_";
    else if (tensorExps[e].tensor == outTensor)
      llvm::dbgs() << "output_";
    llvm::dbgs() << "tensor_" << tensorExps[e].tensor;
    break;
  case kInvariant:
    llvm::dbgs() << "invariant";
    break;
  case kLoopVar:
    llvm::dbgs() << "loopvar_" << tensorExps[e].loop;
    break;
  // Unary operations.
  case kAbsF:
  case kAbsC:
  case kAbsI:
  case kCeilF:
  case kFloorF:
  case kSqrtF:
  case kSqrtC:
  case kExpm1F:
  case kExpm1C:
  case kLog1pF:
  case kLog1pC:
  case kSinF:
  case kSinC:
  case kTanhF:
  case kTanhC:
  case kNegF:
  case kNegC:
  case kNegI:
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kCIm:
  case kCRe:
  case kBitCast:
  case kBinaryBranch:
  case kUnary:
  case kSelect:
    llvm::dbgs() << kindToOpSymbol(tensorExps[e].kind) << " ";
    dumpExp(tensorExps[e].children.e0);
    break;
  // Binary operations.
  case kMulF:
  case kMulC:
  case kMulI:
  case kDivF:
  case kDivC:
  case kDivS:
  case kDivU:
  case kAddF:
  case kAddC:
  case kAddI:
  case kSubF:
  case kSubC:
  case kSubI:
  case kAndI:
  case kOrI:
  case kXorI:
  case kShrS:
  case kShrU:
  case kShlI:
  case kBinary:
  case kReduce:
    llvm::dbgs() << "(";
    dumpExp(tensorExps[e].children.e0);
    llvm::dbgs() << " " << kindToOpSymbol(tensorExps[e].kind) << " ";
    dumpExp(tensorExps[e].children.e1);
    llvm::dbgs() << ")";
  }
}

void Merger::dumpLat(LatPointId p) const {
  llvm::dbgs() << "lat(";
  dumpBits(latPoints[p].bits);
  llvm::dbgs() << " :";
  dumpBits(latPoints[p].simple);
  llvm::dbgs() << " : ";
  dumpExp(latPoints[p].exp);
  llvm::dbgs() << " )\n";
}

void Merger::dumpSet(LatSetId s) const {
  llvm::dbgs() << "{ #" << latSets[s].size() << "\n";
  for (const LatPointId p : latSets[s]) {
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
      llvm::dbgs() << " i_" << t << "_" << i << "_" << toMLIRString(dlt);
    }
  }
}

#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Builder methods.
//===----------------------------------------------------------------------===//

LatSetId Merger::buildLattices(ExprId e, LoopId i) {
  const Kind kind = tensorExps[e].kind;
  switch (kind) {
  // Leaf.
  case kTensor:
  case kInvariant:
  case kLoopVar: {
    // Either the loop-var is really used in the tensor expression, or it is
    // set to the undefined loop-var in that level. An invariant expression,
    // a proper index value, and a truly dynamic sparse output tensor are set
    // to a synthetic tensor with undefined indices only to ensure the
    // iteration space is not skipped as a result of their contents.
    const LatSetId s = addSet();
    TensorId t = syntheticTensor;
    if (kind == kTensor) {
      t = tensorExps[e].tensor;
      if (hasSparseOut && t == outTensor)
        t = syntheticTensor;
    }
    latSets[s].push_back(addLat(t, i, e));
    return s;
  }
  // Unary operations.
  case kAbsF:
  case kAbsC:
  case kAbsI:
  case kCeilF:
  case kFloorF:
  case kSqrtF:
  case kSqrtC:
  case kExpm1F:
  case kExpm1C:
  case kLog1pF:
  case kLog1pC:
  case kSinF:
  case kSinC:
  case kTanhF:
  case kTanhC:
  case kNegF:
  case kNegC:
  case kNegI:
  case kTruncF:
  case kExtF:
  case kCastFS:
  case kCastFU:
  case kCastSF:
  case kCastUF:
  case kCastS:
  case kCastU:
  case kCastIdx:
  case kTruncI:
  case kCIm:
  case kCRe:
  case kBitCast:
    // A zero preserving operation (viz. f(0) = 0, [Bik96,Ch5]) maps the
    // lattice set of the operand through the operator into a new set.
    //
    //  -y|!y | y |
    //  --+---+---+
    //    | 0 |-y |
    return mapSet(kind, buildLattices(tensorExps[e].children.e0, i),
                  tensorExps[e].val);
  case kBinaryBranch:
  case kSelect:
    // The left or right half of a binary operation which has already
    // been split into separate operations for each region.
    return mapSet(kind, buildLattices(tensorExps[e].children.e0, i), Value(),
                  tensorExps[e].op);
  case kUnary:
    // A custom unary operation.
    //
    //  op y|    !y    |     y      |
    //  ----+----------+------------+
    //      | absent() | present(y) |
    {
      const LatSetId child0 = buildLattices(tensorExps[e].children.e0, i);
      UnaryOp unop = cast<UnaryOp>(tensorExps[e].op);
      Region &absentRegion = unop.getAbsentRegion();

      if (absentRegion.empty()) {
        // Simple mapping over existing values.
        return mapSet(kind, child0, Value(), unop);
      } // Use a disjunction with `unop` on the left and the absent value as an
      // invariant on the right.
      Block &absentBlock = absentRegion.front();
      YieldOp absentYield = cast<YieldOp>(absentBlock.getTerminator());
      Value absentVal = absentYield.getResult();
      const ExprId rhs = addExp(kInvariant, absentVal);
      return disjSet(kind, child0, buildLattices(rhs, i), unop);
    }
  // Binary operations.
  case kMulF:
  case kMulC:
  case kMulI:
  case kAndI:
    // A multiplicative operation only needs to be performed
    // for the conjunction of sparse iteration spaces.
    //
    //  x*y|!y | y |
    //  ---+---+---+
    //  !x | 0 | 0 |
    //   x | 0 |x*y|
    //
    // Note even here, 0*NaN=NaN and 0*Inf=NaN, but that is ignored.
    return conjSet(kind, buildLattices(tensorExps[e].children.e0, i),
                   buildLattices(tensorExps[e].children.e1, i));
  case kDivF:
  case kDivC:
  case kDivS:
  case kDivU:
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
    assert(!maybeZero(tensorExps[e].children.e1));
    return conjSet(kind, buildLattices(tensorExps[e].children.e0, i),
                   buildLattices(tensorExps[e].children.e1, i));
  case kAddF:
  case kAddC:
  case kAddI:
  case kSubF:
  case kSubC:
  case kSubI:
  case kOrI:
  case kXorI:
    // An additive operation needs to be performed
    // for the disjunction of sparse iteration spaces.
    //
    //  x+y|!y | y |    x-y|!y | y |
    //  ---+---+---+    ---+---+---+
    //  !x | 0 | y |    !x | 0 |-y |
    //   x | x |x+y|     x | x |x-y|
    return disjSet(kind, buildLattices(tensorExps[e].children.e0, i),
                   buildLattices(tensorExps[e].children.e1, i));
  case kShrS:
  case kShrU:
  case kShlI:
    // A shift operation by an invariant amount (viz. tensor expressions
    // can only occur at the left-hand-side of the operator) can be handled
    // with the conjuction rule.
    assert(isInvariant(tensorExps[e].children.e1));
    return conjSet(kind, buildLattices(tensorExps[e].children.e0, i),
                   buildLattices(tensorExps[e].children.e1, i));
  case kBinary:
    // A custom binary operation.
    //
    //  x op y|   !y    |       y      |
    //  ------+---------+--------------+
    //    !x  |  empty  |   right(y)   |
    //     x  | left(x) | overlap(x,y) |
    {
      const LatSetId child0 = buildLattices(tensorExps[e].children.e0, i);
      const LatSetId child1 = buildLattices(tensorExps[e].children.e1, i);
      BinaryOp binop = cast<BinaryOp>(tensorExps[e].op);
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
      return combiSet(kBinary, child0, child1, binop, includeLeft,
                      kBinaryBranch, leftYield, includeRight, kBinaryBranch,
                      rightYield);
    }
  case kReduce:
    // A custom reduce operation.
    return conjSet(kind, buildLattices(tensorExps[e].children.e0, i),
                   buildLattices(tensorExps[e].children.e1, i),
                   tensorExps[e].op);
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
  if (tensorExps[e].kind == kInvariant) {
    if (auto c = tensorExps[e].val.getDefiningOp<complex::ConstantOp>()) {
      ArrayAttr arrayAttr = c.getValue();
      return arrayAttr[0].cast<FloatAttr>().getValue().isZero() &&
             arrayAttr[1].cast<FloatAttr>().getValue().isZero();
    }
    if (auto c = tensorExps[e].val.getDefiningOp<arith::ConstantIntOp>())
      return c.value() == 0;
    if (auto c = tensorExps[e].val.getDefiningOp<arith::ConstantFloatOp>())
      return c.value().isZero();
  }
  return true;
}

bool Merger::isInvariant(ExprId e) const {
  return tensorExps[e].kind == kInvariant;
}

Type Merger::inferType(ExprId e, Value src) const {
  // Obtain the destination type from the cast node.
  Type dtp = tensorExps[e].val.getType();
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
    const TensorId argN = arg.getArgNumber();
    // Any argument of the generic op that is not marked as a scalar
    // argument is considered a tensor, indexed by the implicit loop
    // bounds. This includes rank-0 tensor arguments.
    if (arg.getOwner()->getParentOp() == op) {
      OpOperand &t = op->getOpOperand(argN);
      if (!op.isScalar(&t))
        return addExp(kTensor, argN);
      v = t.get(); // get scalar value
    }
    // Any other argument (marked as scalar argument for the generic op
    // or belonging to an enveloping op) is considered invariant.
    return addExp(kInvariant, v);
  }
  // Something defined outside is invariant.
  Operation *def = v.getDefiningOp();
  if (def->getBlock() != &op.getRegion().front())
    return addExp(kInvariant, v);
  // Construct index operations.
  if (def->getNumOperands() == 0) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return addExp(kLoopVar, indexOp.getDim());
  }
  // Construct unary operations if subexpression can be built.
  if (def->getNumOperands() == 1) {
    const auto x = buildTensorExp(op, def->getOperand(0));
    if (x.has_value()) {
      const ExprId e = *x;
      if (isa<math::AbsFOp>(def))
        return addExp(kAbsF, e);
      if (isa<complex::AbsOp>(def))
        return addExp(kAbsC, e);
      if (isa<math::AbsIOp>(def))
        return addExp(kAbsI, e);
      if (isa<math::CeilOp>(def))
        return addExp(kCeilF, e);
      if (isa<math::FloorOp>(def))
        return addExp(kFloorF, e);
      if (isa<math::SqrtOp>(def))
        return addExp(kSqrtF, e);
      if (isa<complex::SqrtOp>(def))
        return addExp(kSqrtC, e);
      if (isa<math::ExpM1Op>(def))
        return addExp(kExpm1F, e);
      if (isa<complex::Expm1Op>(def))
        return addExp(kExpm1C, e);
      if (isa<math::Log1pOp>(def))
        return addExp(kLog1pF, e);
      if (isa<complex::Log1pOp>(def))
        return addExp(kLog1pC, e);
      if (isa<math::SinOp>(def))
        return addExp(kSinF, e);
      if (isa<complex::SinOp>(def))
        return addExp(kSinC, e);
      if (isa<math::TanhOp>(def))
        return addExp(kTanhF, e);
      if (isa<complex::TanhOp>(def))
        return addExp(kTanhC, e);
      if (isa<arith::NegFOp>(def))
        return addExp(kNegF, e); // no negi in std
      if (isa<complex::NegOp>(def))
        return addExp(kNegC, e);
      if (isa<arith::TruncFOp>(def))
        return addExp(kTruncF, e, v);
      if (isa<arith::ExtFOp>(def))
        return addExp(kExtF, e, v);
      if (isa<arith::FPToSIOp>(def))
        return addExp(kCastFS, e, v);
      if (isa<arith::FPToUIOp>(def))
        return addExp(kCastFU, e, v);
      if (isa<arith::SIToFPOp>(def))
        return addExp(kCastSF, e, v);
      if (isa<arith::UIToFPOp>(def))
        return addExp(kCastUF, e, v);
      if (isa<arith::ExtSIOp>(def))
        return addExp(kCastS, e, v);
      if (isa<arith::ExtUIOp>(def))
        return addExp(kCastU, e, v);
      if (isa<arith::IndexCastOp>(def))
        return addExp(kCastIdx, e, v);
      if (isa<arith::TruncIOp>(def))
        return addExp(kTruncI, e, v);
      if (isa<complex::ImOp>(def))
        return addExp(kCIm, e);
      if (isa<complex::ReOp>(def))
        return addExp(kCRe, e);
      if (isa<arith::BitcastOp>(def))
        return addExp(kBitCast, e, v);
      if (auto unop = dyn_cast<sparse_tensor::UnaryOp>(def)) {
        if (isAdmissibleBranch(unop, unop.getPresentRegion()) &&
            isAdmissibleBranch(unop, unop.getAbsentRegion()))
          return addExp(kUnary, e, Value(), def);
      }
      if (auto selop = dyn_cast<sparse_tensor::SelectOp>(def)) {
        if (isAdmissibleBranch(selop, selop.getRegion()))
          return addExp(kSelect, e, Value(), def);
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
        return addExp(kMulF, e0, e1);
      if (isa<complex::MulOp>(def))
        return addExp(kMulC, e0, e1);
      if (isa<arith::MulIOp>(def))
        return addExp(kMulI, e0, e1);
      if (isa<arith::DivFOp>(def) && !maybeZero(e1))
        return addExp(kDivF, e0, e1);
      if (isa<complex::DivOp>(def) && !maybeZero(e1))
        return addExp(kDivC, e0, e1);
      if (isa<arith::DivSIOp>(def) && !maybeZero(e1))
        return addExp(kDivS, e0, e1);
      if (isa<arith::DivUIOp>(def) && !maybeZero(e1))
        return addExp(kDivU, e0, e1);
      if (isa<arith::AddFOp>(def))
        return addExp(kAddF, e0, e1);
      if (isa<complex::AddOp>(def))
        return addExp(kAddC, e0, e1);
      if (isa<arith::AddIOp>(def))
        return addExp(kAddI, e0, e1);
      if (isa<arith::SubFOp>(def))
        return addExp(kSubF, e0, e1);
      if (isa<complex::SubOp>(def))
        return addExp(kSubC, e0, e1);
      if (isa<arith::SubIOp>(def))
        return addExp(kSubI, e0, e1);
      if (isa<arith::AndIOp>(def))
        return addExp(kAndI, e0, e1);
      if (isa<arith::OrIOp>(def))
        return addExp(kOrI, e0, e1);
      if (isa<arith::XOrIOp>(def))
        return addExp(kXorI, e0, e1);
      if (isa<arith::ShRSIOp>(def) && isInvariant(e1))
        return addExp(kShrS, e0, e1);
      if (isa<arith::ShRUIOp>(def) && isInvariant(e1))
        return addExp(kShrU, e0, e1);
      if (isa<arith::ShLIOp>(def) && isInvariant(e1))
        return addExp(kShlI, e0, e1);
      if (auto binop = dyn_cast<sparse_tensor::BinaryOp>(def)) {
        if (isAdmissibleBranch(binop, binop.getOverlapRegion()) &&
            (binop.getLeftIdentity() ||
             isAdmissibleBranch(binop, binop.getLeftRegion())) &&
            (binop.getRightIdentity() ||
             isAdmissibleBranch(binop, binop.getRightRegion())))
          return addExp(kBinary, e0, e1, Value(), def);
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
          return addExp(kReduce, e0, e1, Value(), def);
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
  switch (tensorExps[e].kind) {
  // Leaf.
  case kTensor:
  case kInvariant:
  case kLoopVar:
    llvm_unreachable("unexpected non-op");
  // Unary operations.
  case kAbsF:
    return rewriter.create<math::AbsFOp>(loc, v0);
  case kAbsC: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::AbsOp>(loc, eltType, v0);
  }
  case kAbsI:
    return rewriter.create<math::AbsIOp>(loc, v0);
  case kCeilF:
    return rewriter.create<math::CeilOp>(loc, v0);
  case kFloorF:
    return rewriter.create<math::FloorOp>(loc, v0);
  case kSqrtF:
    return rewriter.create<math::SqrtOp>(loc, v0);
  case kSqrtC:
    return rewriter.create<complex::SqrtOp>(loc, v0);
  case kExpm1F:
    return rewriter.create<math::ExpM1Op>(loc, v0);
  case kExpm1C:
    return rewriter.create<complex::Expm1Op>(loc, v0);
  case kLog1pF:
    return rewriter.create<math::Log1pOp>(loc, v0);
  case kLog1pC:
    return rewriter.create<complex::Log1pOp>(loc, v0);
  case kSinF:
    return rewriter.create<math::SinOp>(loc, v0);
  case kSinC:
    return rewriter.create<complex::SinOp>(loc, v0);
  case kTanhF:
    return rewriter.create<math::TanhOp>(loc, v0);
  case kTanhC:
    return rewriter.create<complex::TanhOp>(loc, v0);
  case kNegF:
    return rewriter.create<arith::NegFOp>(loc, v0);
  case kNegC:
    return rewriter.create<complex::NegOp>(loc, v0);
  case kNegI: // no negi in std
    return rewriter.create<arith::SubIOp>(
        loc,
        rewriter.create<arith::ConstantOp>(loc, v0.getType(),
                                           rewriter.getZeroAttr(v0.getType())),
        v0);
  case kTruncF:
    return rewriter.create<arith::TruncFOp>(loc, inferType(e, v0), v0);
  case kExtF:
    return rewriter.create<arith::ExtFOp>(loc, inferType(e, v0), v0);
  case kCastFS:
    return rewriter.create<arith::FPToSIOp>(loc, inferType(e, v0), v0);
  case kCastFU:
    return rewriter.create<arith::FPToUIOp>(loc, inferType(e, v0), v0);
  case kCastSF:
    return rewriter.create<arith::SIToFPOp>(loc, inferType(e, v0), v0);
  case kCastUF:
    return rewriter.create<arith::UIToFPOp>(loc, inferType(e, v0), v0);
  case kCastS:
    return rewriter.create<arith::ExtSIOp>(loc, inferType(e, v0), v0);
  case kCastU:
    return rewriter.create<arith::ExtUIOp>(loc, inferType(e, v0), v0);
  case kCastIdx:
    return rewriter.create<arith::IndexCastOp>(loc, inferType(e, v0), v0);
  case kTruncI:
    return rewriter.create<arith::TruncIOp>(loc, inferType(e, v0), v0);
  case kCIm: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::ImOp>(loc, eltType, v0);
  }
  case kCRe: {
    auto type = v0.getType().cast<ComplexType>();
    auto eltType = type.getElementType().cast<FloatType>();
    return rewriter.create<complex::ReOp>(loc, eltType, v0);
  }
  case kBitCast:
    return rewriter.create<arith::BitcastOp>(loc, inferType(e, v0), v0);
  // Binary operations.
  case kMulF:
    return rewriter.create<arith::MulFOp>(loc, v0, v1);
  case kMulC:
    return rewriter.create<complex::MulOp>(loc, v0, v1);
  case kMulI:
    return rewriter.create<arith::MulIOp>(loc, v0, v1);
  case kDivF:
    return rewriter.create<arith::DivFOp>(loc, v0, v1);
  case kDivC:
    return rewriter.create<complex::DivOp>(loc, v0, v1);
  case kDivS:
    return rewriter.create<arith::DivSIOp>(loc, v0, v1);
  case kDivU:
    return rewriter.create<arith::DivUIOp>(loc, v0, v1);
  case kAddF:
    return rewriter.create<arith::AddFOp>(loc, v0, v1);
  case kAddC:
    return rewriter.create<complex::AddOp>(loc, v0, v1);
  case kAddI:
    return rewriter.create<arith::AddIOp>(loc, v0, v1);
  case kSubF:
    return rewriter.create<arith::SubFOp>(loc, v0, v1);
  case kSubC:
    return rewriter.create<complex::SubOp>(loc, v0, v1);
  case kSubI:
    return rewriter.create<arith::SubIOp>(loc, v0, v1);
  case kAndI:
    return rewriter.create<arith::AndIOp>(loc, v0, v1);
  case kOrI:
    return rewriter.create<arith::OrIOp>(loc, v0, v1);
  case kXorI:
    return rewriter.create<arith::XOrIOp>(loc, v0, v1);
  case kShrS:
    return rewriter.create<arith::ShRSIOp>(loc, v0, v1);
  case kShrU:
    return rewriter.create<arith::ShRUIOp>(loc, v0, v1);
  case kShlI:
    return rewriter.create<arith::ShLIOp>(loc, v0, v1);
  case kBinaryBranch: // semi-ring ops with custom logic.
    return insertYieldOp(rewriter, loc,
                         *tensorExps[e].op->getBlock()->getParent(), {v0});
  case kUnary:
    return buildUnaryPresent(rewriter, loc, tensorExps[e].op, v0);
  case kSelect:
    return insertYieldOp(rewriter, loc,
                         cast<SelectOp>(tensorExps[e].op).getRegion(), {v0});
  case kBinary:
    return buildBinaryOverlap(rewriter, loc, tensorExps[e].op, v0, v1);
  case kReduce: {
    ReduceOp redOp = cast<ReduceOp>(tensorExps[e].op);
    return insertYieldOp(rewriter, loc, redOp.getRegion(), {v0, v1});
  }
  }
  llvm_unreachable("unexpected expression kind in build");
}

} // namespace sparse_tensor
} // namespace mlir
