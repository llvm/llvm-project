//===- Merger.cpp - Implementation of iteration lattices ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/Utils/Merger.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sparse_tensor {

//
// Constructors.
//

TensorExp::TensorExp(Kind k, unsigned x, unsigned y, Value v)
    : kind(k), val(v) {
  switch (kind) {
  case kTensor:
    assert(x != -1u && y == -1u && !v);
    tensor = x;
    break;
  case kInvariant:
    assert(x == -1u && y == -1u && v);
    break;
  case kAbsF:
  case kCeilF:
  case kFloorF:
  case kNegF:
  case kNegI:
    assert(x != -1u && y == -1u && !v);
    children.e0 = x;
    children.e1 = y;
    break;
  default:
    assert(x != -1u && y != -1u && !v);
    children.e0 = x;
    children.e1 = y;
    break;
  }
}

LatPoint::LatPoint(unsigned n, unsigned e, unsigned b)
    : bits(n, false), simple(), exp(e) {
  bits.set(b);
}

LatPoint::LatPoint(const llvm::BitVector &b, unsigned e)
    : bits(b), simple(), exp(e) {}

//
// Lattice methods.
//

unsigned Merger::addExp(Kind k, unsigned e0, unsigned e1, Value v) {
  unsigned e = tensorExps.size();
  tensorExps.push_back(TensorExp(k, e0, e1, v));
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

unsigned Merger::conjLatPoint(Kind kind, unsigned p0, unsigned p1) {
  unsigned p = latPoints.size();
  llvm::BitVector nb = llvm::BitVector(latPoints[p0].bits);
  nb |= latPoints[p1].bits;
  unsigned e = addExp(kind, latPoints[p0].exp, latPoints[p1].exp);
  latPoints.push_back(LatPoint(nb, e));
  return p;
}

unsigned Merger::takeConj(Kind kind, unsigned s0, unsigned s1) {
  unsigned s = addSet();
  for (unsigned p0 : latSets[s0])
    for (unsigned p1 : latSets[s1])
      latSets[s].push_back(conjLatPoint(kind, p0, p1));
  return s;
}

unsigned Merger::takeDisj(Kind kind, unsigned s0, unsigned s1) {
  unsigned s = takeConj(kind, s0, s1);
  // Followed by all in s0.
  for (unsigned p : latSets[s0])
    latSets[s].push_back(p);
  // Map binary 0-y to unary -y.
  if (kind == kSubF)
    s1 = mapSet(kNegF, s1);
  else if (kind == kSubI)
    s1 = mapSet(kNegI, s1);
  // Followed by all in s1.
  for (unsigned p : latSets[s1])
    latSets[s].push_back(p);
  return s;
}

unsigned Merger::mapSet(Kind kind, unsigned s0) {
  assert(kAbsF <= kind && kind <= kNegI);
  unsigned s = addSet();
  for (unsigned p : latSets[s0]) {
    unsigned e = addExp(kind, latPoints[p].exp);
    latPoints.push_back(LatPoint(latPoints[p].bits, e));
    latSets[s].push_back(latPoints.size() - 1);
  }
  return s;
}

unsigned Merger::optimizeSet(unsigned s0) {
  unsigned s = addSet();
  assert(latSets[s0].size() != 0);
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

llvm::BitVector Merger::simplifyCond(unsigned s0, unsigned p0) {
  // First determine if this lattice point is a *singleton*, i.e.,
  // the last point in a lattice, no other is less than this one.
  bool isSingleton = true;
  for (unsigned p1 : latSets[s0]) {
    if (p0 != p1 && latGT(p0, p1)) {
      isSingleton = false;
      break;
    }
  }
  // Now apply the two basic rules.
  llvm::BitVector simple = latPoints[p0].bits;
  bool reset = isSingleton && hasAnyDimOf(simple, kSparse);
  for (unsigned b = 0, be = simple.size(); b < be; b++) {
    if (simple[b] && !isDim(b, kSparse)) {
      if (reset)
        simple.reset(b);
      reset = true;
    }
  }
  return simple;
}

bool Merger::latGT(unsigned i, unsigned j) const {
  const llvm::BitVector &bitsi = latPoints[i].bits;
  const llvm::BitVector &bitsj = latPoints[j].bits;
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
  llvm::BitVector tmp = latPoints[j].bits;
  tmp ^= latPoints[i].bits;
  return !hasAnyDimOf(tmp, kSparse);
}

bool Merger::hasAnyDimOf(const llvm::BitVector &bits, Dim d) const {
  for (unsigned b = 0, be = bits.size(); b < be; b++)
    if (bits[b] && isDim(b, d))
      return true;
  return false;
}

bool Merger::isConjunction(unsigned t, unsigned e) const {
  switch (tensorExps[e].kind) {
  case kTensor:
    return tensorExps[e].tensor == t;
  case kAbsF:
  case kCeilF:
  case kFloorF:
  case kNegF:
  case kNegI:
    return isConjunction(t, tensorExps[e].children.e0);
  case kDivF: // note: x / c only
  case kDivS:
  case kDivU:
    assert(!maybeZero(tensorExps[e].children.e1));
    return isConjunction(t, tensorExps[e].children.e0);
  case kShrS: // note: x >> inv only
  case kShrU:
  case kShlI:
    assert(isInvariant(tensorExps[e].children.e1));
    return isConjunction(t, tensorExps[e].children.e0);
  case kMulF:
  case kMulI:
  case kAndI:
    return isConjunction(t, tensorExps[e].children.e0) ||
           isConjunction(t, tensorExps[e].children.e1);
  default:
    return false;
  }
}

#ifndef NDEBUG

//
// Print methods (for debugging).
//

static const char *kindToOpSymbol(Kind kind) {
  switch (kind) {
  case kTensor:
    return "tensor";
  case kInvariant:
    return "invariant";
  case kAbsF:
    return "abs";
  case kCeilF:
    return "ceil";
  case kFloorF:
    return "floor";
  case kNegF:
    return "-";
  case kNegI:
    return "-";
  case kMulF:
    return "*";
  case kMulI:
    return "*";
  case kDivF:
    return "/";
  case kDivS:
    return "/";
  case kDivU:
    return "/";
  case kAddF:
    return "+";
  case kAddI:
    return "+";
  case kSubF:
    return "-";
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
  }
  llvm_unreachable("unexpected kind for symbol");
}

void Merger::dumpExp(unsigned e) const {
  switch (tensorExps[e].kind) {
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
  case kAbsF:
  case kCeilF:
  case kFloorF:
  case kNegF:
  case kNegI:
    llvm::dbgs() << kindToOpSymbol(tensorExps[e].kind) << " ";
    dumpExp(tensorExps[e].children.e0);
    break;
  default:
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

void Merger::dumpBits(const llvm::BitVector &bits) const {
  for (unsigned b = 0, be = bits.size(); b < be; b++) {
    if (bits[b]) {
      unsigned t = tensor(b);
      unsigned i = index(b);
      llvm::dbgs() << " i_" << t << "_" << i << "_";
      switch (dims[t][i]) {
      case kSparse:
        llvm::dbgs() << "S";
        break;
      case kDense:
        llvm::dbgs() << "D";
        break;
      case kSingle:
        llvm::dbgs() << "T";
        break;
      case kUndef:
        llvm::dbgs() << "U";
        break;
      }
    }
  }
}

#endif // NDEBUG

//
// Builder methods.
//

unsigned Merger::buildLattices(unsigned e, unsigned i) {
  Kind kind = tensorExps[e].kind;
  switch (kind) {
  case kTensor:
  case kInvariant: {
    // Either the index is really used in the tensor expression, or it is
    // set to the undefined index in that dimension. An invariant expression
    // is set to a synthetic tensor with undefined indices only.
    unsigned s = addSet();
    unsigned t = kind == kTensor ? tensorExps[e].tensor : syntheticTensor;
    latSets[s].push_back(addLat(t, i, e));
    return s;
  }
  case kAbsF:
  case kCeilF:
  case kFloorF:
  case kNegF:
  case kNegI:
    // A zero preserving operation (viz. f(0) = 0, [Bik96,Ch5]) maps the
    // lattice set of the operand through the operator into a new set.
    //
    //  -y|!y | y |
    //  --+---+---+
    //    | 0 |-y |
    return mapSet(kind, buildLattices(tensorExps[e].children.e0, i));
  case kMulF:
  case kMulI:
  case kAndI:
    // A multiplicative operation only needs to be performed
    // for the conjunction of sparse iteration spaces.
    //
    //  x*y|!y | y |
    //  ---+---+---+
    //  !x | 0 | 0 |
    //   x | 0 |x*y|
    return takeConj(kind, // take binary conjunction
                    buildLattices(tensorExps[e].children.e0, i),
                    buildLattices(tensorExps[e].children.e1, i));
  case kDivF:
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
  case kAddI:
  case kSubF:
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
  }
  llvm_unreachable("unexpected expression kind");
}

Optional<unsigned> Merger::buildTensorExpFromLinalg(linalg::GenericOp op) {
  Operation *yield = op.region().front().getTerminator();
  return buildTensorExp(op, yield->getOperand(0));
}

bool Merger::maybeZero(unsigned e) const {
  if (tensorExps[e].kind == kInvariant) {
    if (auto c = tensorExps[e].val.getDefiningOp<ConstantIntOp>())
      return c.getValue() == 0;
    if (auto c = tensorExps[e].val.getDefiningOp<ConstantFloatOp>())
      return c.getValue().isZero();
  }
  return true;
}

bool Merger::isInvariant(unsigned e) const {
  return tensorExps[e].kind == kInvariant;
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
  if (def->getBlock() != &op.region().front())
    return addExp(kInvariant, v);
  // Construct unary operations if subexpression can be built.
  if (def->getNumOperands() == 1) {
    auto x = buildTensorExp(op, def->getOperand(0));
    if (x.hasValue()) {
      unsigned e = x.getValue();
      if (isa<AbsFOp>(def))
        return addExp(kAbsF, e);
      if (isa<CeilFOp>(def))
        return addExp(kCeilF, e);
      if (isa<FloorFOp>(def))
        return addExp(kFloorF, e);
      if (isa<NegFOp>(def))
        return addExp(kNegF, e);
      // TODO: no negi in std?
    }
  }
  // Construct binary operations if subexpressions can be built.
  // TODO: see buildLattices() for an explanation of rejecting certain divisions
  if (def->getNumOperands() == 2) {
    auto x = buildTensorExp(op, def->getOperand(0));
    auto y = buildTensorExp(op, def->getOperand(1));
    if (x.hasValue() && y.hasValue()) {
      unsigned e0 = x.getValue();
      unsigned e1 = y.getValue();
      if (isa<MulFOp>(def))
        return addExp(kMulF, e0, e1);
      if (isa<MulIOp>(def))
        return addExp(kMulI, e0, e1);
      if (isa<DivFOp>(def) && !maybeZero(e1))
        return addExp(kDivF, e0, e1);
      if (isa<SignedDivIOp>(def) && !maybeZero(e1))
        return addExp(kDivS, e0, e1);
      if (isa<UnsignedDivIOp>(def) && !maybeZero(e1))
        return addExp(kDivU, e0, e1);
      if (isa<AddFOp>(def))
        return addExp(kAddF, e0, e1);
      if (isa<AddIOp>(def))
        return addExp(kAddI, e0, e1);
      if (isa<SubFOp>(def))
        return addExp(kSubF, e0, e1);
      if (isa<SubIOp>(def))
        return addExp(kSubI, e0, e1);
      if (isa<AndOp>(def))
        return addExp(kAndI, e0, e1);
      if (isa<OrOp>(def))
        return addExp(kOrI, e0, e1);
      if (isa<XOrOp>(def))
        return addExp(kXorI, e0, e1);
      if (isa<SignedShiftRightOp>(def) && isInvariant(e1))
        return addExp(kShrS, e0, e1);
      if (isa<UnsignedShiftRightOp>(def) && isInvariant(e1))
        return addExp(kShrU, e0, e1);
      if (isa<ShiftLeftOp>(def) && isInvariant(e1))
        return addExp(kShlI, e0, e1);
    }
  }
  // Cannot build.
  return None;
}

Value Merger::buildExp(PatternRewriter &rewriter, Location loc, unsigned e,
                       Value v0, Value v1) {
  switch (tensorExps[e].kind) {
  case kTensor:
  case kInvariant:
    llvm_unreachable("unexpected non-op");
  case kAbsF:
    return rewriter.create<AbsFOp>(loc, v0);
  case kCeilF:
    return rewriter.create<CeilFOp>(loc, v0);
  case kFloorF:
    return rewriter.create<FloorFOp>(loc, v0);
  case kNegF:
    return rewriter.create<NegFOp>(loc, v0);
  case kNegI:
    assert(v1); // no negi in std
    return rewriter.create<SubIOp>(loc, v0, v1);
  case kMulF:
    return rewriter.create<MulFOp>(loc, v0, v1);
  case kMulI:
    return rewriter.create<MulIOp>(loc, v0, v1);
  case kDivF:
    return rewriter.create<DivFOp>(loc, v0, v1);
  case kDivS:
    return rewriter.create<SignedDivIOp>(loc, v0, v1);
  case kDivU:
    return rewriter.create<UnsignedDivIOp>(loc, v0, v1);
  case kAddF:
    return rewriter.create<AddFOp>(loc, v0, v1);
  case kAddI:
    return rewriter.create<AddIOp>(loc, v0, v1);
  case kSubF:
    return rewriter.create<SubFOp>(loc, v0, v1);
  case kSubI:
    return rewriter.create<SubIOp>(loc, v0, v1);
  case kAndI:
    return rewriter.create<AndOp>(loc, v0, v1);
  case kOrI:
    return rewriter.create<OrOp>(loc, v0, v1);
  case kXorI:
    return rewriter.create<XOrOp>(loc, v0, v1);
  case kShrS:
    return rewriter.create<SignedShiftRightOp>(loc, v0, v1);
  case kShrU:
    return rewriter.create<UnsignedShiftRightOp>(loc, v0, v1);
  case kShlI:
    return rewriter.create<ShiftLeftOp>(loc, v0, v1);
  }
  llvm_unreachable("unexpected expression kind in build");
}

} // namespace sparse_tensor
} // namespace mlir
