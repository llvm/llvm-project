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
  for (unsigned p : latSets[s0])
    latSets[s].push_back(p);
  for (unsigned p : latSets[s1])
    latSets[s].push_back(p);
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
      if (exp(e).kind == Kind::kTensor && exp(e).e0 == outTensor)
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

llvm::BitVector Merger::simplifyCond(unsigned s, unsigned p0) {
  // First determine if this lattice point is a *singleton*, i.e.,
  // the last point in a lattice, no other is less than this one.
  bool isSingleton = true;
  for (unsigned p1 : latSets[s]) {
    if (p0 != p1 && latGT(p0, p1)) {
      isSingleton = false;
      break;
    }
  }
  // Now apply the two basic rules.
  llvm::BitVector simple = latPoints[p0].bits;
  bool reset = isSingleton && hasAnyDimOf(simple, Dim::kSparse);
  for (unsigned b = 0, be = simple.size(); b < be; b++) {
    if (simple[b] && !isDim(b, Dim::kSparse)) {
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
  return !hasAnyDimOf(tmp, Dim::kSparse);
}

bool Merger::hasAnyDimOf(const llvm::BitVector &bits, Dim d) const {
  for (unsigned b = 0, be = bits.size(); b < be; b++)
    if (bits[b] && isDim(b, d))
      return true;
  return false;
}

unsigned Merger::buildLattices(unsigned e, unsigned idx) {
  Kind kind = exp(e).kind;
  if (kind == Kind::kTensor || kind == Kind::kInvariant) {
    // Either the index is really used in the tensor expression, or it is
    // set to the undefined index in that dimension. An invariant expression
    // is set to a synthetic tensor with undefined indices only.
    unsigned s = addSet();
    unsigned t = kind == Kind::kTensor ? exp(e).e0 : syntheticTensor;
    set(s).push_back(addLat(t, idx, e));
    return s;
  }
  unsigned s0 = buildLattices(exp(e).e0, idx);
  unsigned s1 = buildLattices(exp(e).e1, idx);
  switch (kind) {
  case Kind::kTensor:
  case Kind::kInvariant:
    llvm_unreachable("handled above");
  case Kind::kMulF:
  case Kind::kMulI:
    return takeConj(kind, s0, s1);
  case Kind::kAddF:
  case Kind::kAddI:
    return takeDisj(kind, s0, s1);
  }
  llvm_unreachable("unexpected expression kind");
}

#ifndef NDEBUG

//
// Print methods (for debugging).
//

void Merger::dumpExp(unsigned e) const {
  switch (tensorExps[e].kind) {
  case Kind::kTensor:
    llvm::dbgs() << "tensor_" << tensorExps[e].e0;
    break;
  case Kind::kInvariant:
    llvm::dbgs() << "invariant";
    break;
  default:
  case Kind::kMulI:
    llvm::dbgs() << "(";
    dumpExp(tensorExps[e].e0);
    llvm::dbgs() << " * ";
    dumpExp(tensorExps[e].e1);
    llvm::dbgs() << ")";
    break;
  case Kind::kAddF:
  case Kind::kAddI:
    llvm::dbgs() << "(";
    dumpExp(tensorExps[e].e0);
    llvm::dbgs() << " + ";
    dumpExp(tensorExps[e].e1);
    llvm::dbgs() << ")";
    break;
  }
}

void Merger::dumpLat(unsigned p) const {
  llvm::dbgs() << "lat(";
  dumpBits(latPoints[p].bits);
  llvm::dbgs() << " :";
  dumpBits(latPoints[p].simple);
  llvm::dbgs() << " / ";
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
      case Dim::kSparse:
        llvm::dbgs() << "S";
        break;
      case Dim::kDense:
        llvm::dbgs() << "D";
        break;
      case Dim::kSingle:
        llvm::dbgs() << "T";
        break;
      case Dim::kUndef:
        llvm::dbgs() << "U";
        break;
      }
    }
  }
}

#endif // NDEBUG

} // namespace sparse_tensor
} // namespace mlir
