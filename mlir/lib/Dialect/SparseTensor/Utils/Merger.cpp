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

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

TensorExp::TensorExp(Kind k, unsigned x, unsigned y, Value v, Operation *o)
    : kind(k), val(v), op(o) {
  switch (kind) {
  // Leaf.
  case kTensor:
    assert(x != -1u && y == -1u && !v && !o);
    tensor = x;
    break;
  case kInvariant:
    assert(x == -1u && y == -1u && v && !o);
    break;
  case kIndex:
    assert(x != -1u && y == -1u && !v && !o);
    index = x;
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
    assert(x != -1u && y == -1u && !v && !o);
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
    assert(x != -1u && y == -1u && v && !o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kBinaryBranch:
  case kSelect:
    assert(x != -1u && y == -1u && !v && o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kUnary:
    // No assertion on y can be made, as the branching paths involve both
    // a unary (mapSet) and binary (takeDisj) pathway.
    assert(x != -1u && !v && o);
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
    assert(x != -1u && y != -1u && !v && !o);
    children.e0 = x;
    children.e1 = y;
    break;
  case kBinary:
  case kReduce:
    assert(x != -1u && y != -1u && !v && o);
    children.e0 = x;
    children.e1 = y;
    break;
  }
}

LatPoint::LatPoint(unsigned n, unsigned e, unsigned b)
    : bits(n, false), exp(e) {
  bits.set(b);
}

LatPoint::LatPoint(const BitVector &b, unsigned e) : bits(b), exp(e) {}

//===----------------------------------------------------------------------===//
// Lattice methods.
//===----------------------------------------------------------------------===//

unsigned Merger::addExp(Kind k, unsigned e0, unsigned e1, Value v,
                        Operation *op) {
  unsigned e = tensorExps.size();
  tensorExps.push_back(TensorExp(k, e0, e1, v, op));
  return e;
}

unsigned Merger::addLat(unsigned t, unsigned i, unsigned e) {
  assert(t < numTensors && i < numLoops);
  unsigned p = latPoints.size();
  latPoints.push_back(LatPoint(numLoops * numTensors, e, numTensors * i + t));
  return p;
}

unsigned Merger::addSet() {
  unsigned s = latSets.size();
  latSets.emplace_back(SmallVector<unsigned, 16>());
  return s;
}

unsigned Merger::conjLatPoint(Kind kind, unsigned p0, unsigned p1,
                              Operation *op) {
  unsigned p = latPoints.size();
  BitVector nb = BitVector(latPoints[p0].bits);
  nb |= latPoints[p1].bits;
  unsigned e = addExp(kind, latPoints[p0].exp, latPoints[p1].exp, Value(), op);
  latPoints.push_back(LatPoint(nb, e));
  return p;
}

unsigned Merger::takeConj(Kind kind, unsigned s0, unsigned s1, Operation *op) {
  unsigned s = addSet();
  for (unsigned p0 : latSets[s0])
    for (unsigned p1 : latSets[s1])
      latSets[s].push_back(conjLatPoint(kind, p0, p1, op));
  return s;
}

unsigned Merger::takeDisj(Kind kind, unsigned s0, unsigned s1, Operation *op) {
  unsigned s = takeConj(kind, s0, s1, op);
  // Followed by all in s0.
  for (unsigned p : latSets[s0])
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
  for (unsigned p : latSets[s1])
    latSets[s].push_back(p);
  return s;
}

unsigned Merger::takeCombi(Kind kind, unsigned s0, unsigned s1, Operation *orig,
                           bool includeLeft, Kind ltrans, Operation *opleft,
                           bool includeRight, Kind rtrans, Operation *opright) {
  unsigned s = takeConj(kind, s0, s1, orig);
  // Left Region.
  if (includeLeft) {
    if (opleft)
      s0 = mapSet(ltrans, s0, Value(), opleft);
    for (unsigned p : latSets[s0])
      latSets[s].push_back(p);
  }
  // Right Region.
  if (includeRight) {
    if (opright)
      s1 = mapSet(rtrans, s1, Value(), opright);
    for (unsigned p : latSets[s1])
      latSets[s].push_back(p);
  }
  return s;
}

unsigned Merger::mapSet(Kind kind, unsigned s0, Value v, Operation *op) {
  assert(kAbsF <= kind && kind <= kSelect);
  unsigned s = addSet();
  for (unsigned p : latSets[s0]) {
    unsigned e = addExp(kind, latPoints[p].exp, v, op);
    latPoints.push_back(LatPoint(latPoints[p].bits, e));
    latSets[s].push_back(latPoints.size() - 1);
  }
  return s;
}

unsigned Merger::optimizeSet(unsigned s0) {
  unsigned s = addSet();
  assert(!latSets[s0].empty());
  unsigned p0 = latSets[s0][0];
  for (unsigned p1 : latSets[s0]) {
    bool add = true;
    if (p0 != p1) {
      // Is this a straightforward copy?
      unsigned e = latPoints[p1].exp;
      if (tensorExps[e].kind == kTensor && tensorExps[e].tensor == outTensor)
        continue;
      // Conjunction already covered?
      for (unsigned p2 : latSets[s]) {
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
  for (unsigned p : latSets[s])
    latPoints[p].simple = simplifyCond(s, p);
  return s;
}

BitVector Merger::simplifyCond(unsigned s0, unsigned p0) {
  // First determine if this lattice point is a *singleton*, i.e.,
  // the last point in a lattice, no other is less than this one.
  bool isSingleton = true;
  for (unsigned p1 : latSets[s0]) {
    if (p0 != p1 && latGT(p0, p1)) {
      isSingleton = false;
      break;
    }
  }

  BitVector simple = latPoints[p0].bits;
  bool reset = isSingleton && hasAnySparse(simple);
  unsigned be = simple.size();
  unsigned offset = 0; // relative to the end
  if (!reset)
    // Starts resetting from a dense dimension, so that the first bit (if kept)
    // is not undefined dimension type.
    for (unsigned b = 0; b < be; b++) {
      if (simple[b] && isDimLevelType(b, DimLvlType::kDense)) {
        offset = be - b - 1; // relative to the end
        break;
      }
    }

  // Now apply the two basic rules. We also iterate the bits reversely to always
  // keep the rightmost bit (which could possibly be a synthetic tensor).
  for (unsigned b = be - 1 - offset, i = 0; i < be;
       b = b == 0 ? be - 1 : b - 1, i++) {
    if (simple[b] && (!isDimLevelType(b, DimLvlType::kCompressed) &&
                      !isDimLevelType(b, DimLvlType::kSingleton))) {
      if (reset)
        simple.reset(b);
      reset = true;
    }
  }
  return simple;
}

bool Merger::latGT(unsigned i, unsigned j) const {
  const BitVector &bitsi = latPoints[i].bits;
  const BitVector &bitsj = latPoints[j].bits;
  assert(bitsi.size() == bitsj.size());
  if (bitsi.count() > bitsj.count()) {
    for (unsigned b = 0, be = bitsj.size(); b < be; b++)
      if (bitsj[b] && !bitsi[b])
        return false;
    return true;
  }
  return false;
}

bool Merger::onlyDenseDiff(unsigned i, unsigned j) {
  BitVector tmp = latPoints[j].bits;
  tmp ^= latPoints[i].bits;
  return !hasAnySparse(tmp);
}

bool Merger::isSingleCondition(unsigned t, unsigned e) const {
  switch (tensorExps[e].kind) {
  // Leaf.
  case kTensor:
    return tensorExps[e].tensor == t;
  case kInvariant:
  case kIndex:
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
  for (unsigned b = 0, be = bits.size(); b < be; b++)
    if (bits[b] && (isDimLevelType(b, DimLvlType::kCompressed) ||
                    isDimLevelType(b, DimLvlType::kSingleton)))
      return true;
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
  case kIndex:
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

void Merger::dumpExp(unsigned e) const {
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
  case kIndex:
    llvm::dbgs() << "index_" << tensorExps[e].index;
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

void Merger::dumpLat(unsigned p) const {
  llvm::dbgs() << "lat(";
  dumpBits(latPoints[p].bits);
  llvm::dbgs() << " :";
  dumpBits(latPoints[p].simple);
  llvm::dbgs() << " : ";
  dumpExp(latPoints[p].exp);
  llvm::dbgs() << " )\n";
}

void Merger::dumpSet(unsigned s) const {
  llvm::dbgs() << "{ #" << latSets[s].size() << "\n";
  for (unsigned p : latSets[s]) {
    llvm::dbgs() << "  ";
    dumpLat(p);
  }
  llvm::dbgs() << "}\n";
}

void Merger::dumpBits(const BitVector &bits) const {
  for (unsigned b = 0, be = bits.size(); b < be; b++) {
    if (bits[b]) {
      unsigned t = tensor(b);
      unsigned i = index(b);
      DimLevelFormat f = dims[t][i];
      llvm::dbgs() << " i_" << t << "_" << i << "_";
      switch (f.levelType) {
      case DimLvlType::kDense:
        llvm::dbgs() << "D";
        break;
      case DimLvlType::kCompressed:
        llvm::dbgs() << "C";
        break;
      case DimLvlType::kSingleton:
        llvm::dbgs() << "S";
        break;
      case DimLvlType::kUndef:
        llvm::dbgs() << "U";
        break;
      }
      llvm::dbgs() << "[O=" << f.isOrdered << ",U=" << f.isUnique << "]";
    }
  }
}

#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Builder methods.
//===----------------------------------------------------------------------===//

unsigned Merger::buildLattices(unsigned e, unsigned i) {
  Kind kind = tensorExps[e].kind;
  switch (kind) {
  // Leaf.
  case kTensor:
  case kInvariant:
  case kIndex: {
    // Either the index is really used in the tensor expression, or it is
    // set to the undefined index in that dimension. An invariant expression,
    // a proper index value, and a truly dynamic sparse output tensor are set
    // to a synthetic tensor with undefined indices only to ensure the
    // iteration space is not skipped as a result of their contents.
    unsigned s = addSet();
    unsigned t = syntheticTensor;
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
      unsigned child0 = buildLattices(tensorExps[e].children.e0, i);
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
      unsigned rhs = addExp(kInvariant, absentVal);
      return takeDisj(kind, child0, buildLattices(rhs, i), unop);
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
    return takeConj(kind, // take binary conjunction
                    buildLattices(tensorExps[e].children.e0, i),
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
    return takeConj(kind, // take binary conjunction
                    buildLattices(tensorExps[e].children.e0, i),
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
    return takeDisj(kind, // take binary disjunction
                    buildLattices(tensorExps[e].children.e0, i),
                    buildLattices(tensorExps[e].children.e1, i));
  case kShrS:
  case kShrU:
  case kShlI:
    // A shift operation by an invariant amount (viz. tensor expressions
    // can only occur at the left-hand-side of the operator) can be handled
    // with the conjuction rule.
    assert(isInvariant(tensorExps[e].children.e1));
    return takeConj(kind, // take binary conjunction
                    buildLattices(tensorExps[e].children.e0, i),
                    buildLattices(tensorExps[e].children.e1, i));
  case kBinary:
    // A custom binary operation.
    //
    //  x op y|   !y    |       y      |
    //  ------+---------+--------------+
    //    !x  |  empty  |   right(y)   |
    //     x  | left(x) | overlap(x,y) |
    {
      unsigned child0 = buildLattices(tensorExps[e].children.e0, i);
      unsigned child1 = buildLattices(tensorExps[e].children.e1, i);
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
      return takeCombi(kBinary, child0, child1, binop, includeLeft,
                       kBinaryBranch, leftYield, includeRight, kBinaryBranch,
                       rightYield);
    }
  case kReduce:
    // A custom reduce operation.
    return takeConj(kind, buildLattices(tensorExps[e].children.e0, i),
                    buildLattices(tensorExps[e].children.e1, i),
                    tensorExps[e].op);
  }
  llvm_unreachable("unexpected expression kind");
}

Optional<unsigned> Merger::buildTensorExpFromLinalg(linalg::GenericOp op) {
  // Build the linalg semantics backward from yield.
  Operation *yield = op.getRegion().front().getTerminator();
  assert(isa<linalg::YieldOp>(yield));
  return buildTensorExp(op, yield->getOperand(0));
}

/// Only returns false if we are certain this is a nonzero.
bool Merger::maybeZero(unsigned e) const {
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

bool Merger::isInvariant(unsigned e) const {
  return tensorExps[e].kind == kInvariant;
}

Type Merger::inferType(unsigned e, Value src) {
  // Obtain the destination type from the cast node.
  Type dtp = tensorExps[e].val.getType();
  // Inspect source type. For vector types, apply the same
  // vectorization to the destination type.
  if (auto vtp = src.getType().dyn_cast<VectorType>())
    return VectorType::get(vtp.getNumElements(), dtp, vtp.getNumScalableDims());
  return dtp;
}

/// Ensures that sparse compiler can generate code for expression.
static bool isAdmissableBranchExp(Operation *op, Block *block, Value v) {
  // Arguments are always admissible.
  if (auto arg = v.dyn_cast<BlockArgument>())
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
    if (!isAdmissableBranchExp(op, block, def->getOperand(i)))
      return false;
  return true;
}

/// Ensures that sparse compiler can generate code for branch.
static bool isAdmissableBranch(Operation *op, Region &region) {
  if (region.empty())
    return true;
  // Build the semi-ring branch semantics backward from yield.
  Operation *yield = region.front().getTerminator();
  assert(isa<YieldOp>(yield));
  return isAdmissableBranchExp(op, &region.front(), yield->getOperand(0));
}

Optional<unsigned> Merger::buildTensorExp(linalg::GenericOp op, Value v) {
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    unsigned argN = arg.getArgNumber();
    // Any argument of the generic op that is not marked as a scalar
    // argument is considered a tensor, indexed by the implicit loop
    // bounds. This includes rank-0 tensor arguments.
    if (arg.getOwner()->getParentOp() == op) {
      OpOperand *t = op.getInputAndOutputOperands()[argN];
      if (!op.isScalar(t))
        return addExp(kTensor, argN);
      v = t->get(); // get scalar value
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
      return addExp(kIndex, indexOp.getDim());
  }
  // Construct unary operations if subexpression can be built.
  if (def->getNumOperands() == 1) {
    auto x = buildTensorExp(op, def->getOperand(0));
    if (x.has_value()) {
      unsigned e = x.value();
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
        if (isAdmissableBranch(unop, unop.getPresentRegion()) &&
            isAdmissableBranch(unop, unop.getAbsentRegion()))
          return addExp(kUnary, e, Value(), def);
      }
      if (auto selop = dyn_cast<sparse_tensor::SelectOp>(def)) {
        if (isAdmissableBranch(selop, selop.getRegion()))
          return addExp(kSelect, e, Value(), def);
      }
    }
  }
  // Construct binary operations if subexpressions can be built.
  // See buildLattices() for an explanation of rejecting certain
  // division and shift operations.
  if (def->getNumOperands() == 2) {
    auto x = buildTensorExp(op, def->getOperand(0));
    auto y = buildTensorExp(op, def->getOperand(1));
    if (x.has_value() && y.has_value()) {
      unsigned e0 = x.value();
      unsigned e1 = y.value();
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
        if (isAdmissableBranch(binop, binop.getOverlapRegion()) &&
            (binop.getLeftIdentity() ||
             isAdmissableBranch(binop, binop.getLeftRegion())) &&
            (binop.getRightIdentity() ||
             isAdmissableBranch(binop, binop.getRightRegion())))
          return addExp(kBinary, e0, e1, Value(), def);
      }
    }
  }
  // Construct ternary operations if subexpressions can be built.
  if (def->getNumOperands() == 3) {
    auto x = buildTensorExp(op, def->getOperand(0));
    auto y = buildTensorExp(op, def->getOperand(1));
    auto z = buildTensorExp(op, def->getOperand(2));
    if (x.has_value() && y.has_value() && z.has_value()) {
      unsigned e0 = x.value();
      unsigned e1 = y.value();
      if (auto redop = dyn_cast<sparse_tensor::ReduceOp>(def)) {
        if (isAdmissableBranch(redop, redop.getRegion()))
          return addExp(kReduce, e0, e1, Value(), def);
      }
    }
  }
  // Cannot build.
  return None;
}

static Value insertYieldOp(RewriterBase &rewriter, Location loc, Region &region,
                           ValueRange vals) {
  // Make a clone of overlap region.
  Region tmpRegion;
  BlockAndValueMapping mapper;
  region.cloneInto(&tmpRegion, tmpRegion.begin(), mapper);
  Block &clonedBlock = tmpRegion.front();
  YieldOp clonedYield = cast<YieldOp>(clonedBlock.getTerminator());
  // Merge cloned block and return yield value.
  Operation *placeholder = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.mergeBlockBefore(&tmpRegion.front(), placeholder, vals);
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

Value Merger::buildExp(RewriterBase &rewriter, Location loc, unsigned e,
                       Value v0, Value v1) {
  switch (tensorExps[e].kind) {
  // Leaf.
  case kTensor:
  case kInvariant:
  case kIndex:
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
