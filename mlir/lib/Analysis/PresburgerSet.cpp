//===- Set.cpp - MLIR PresburgerSet Class ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/PresburgerSet.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

PresburgerSet::PresburgerSet(const FlatAffineConstraints &fac)
    : nDim(fac.getNumDimIds()), nSym(fac.getNumSymbolIds()) {
  unionFACInPlace(fac);
}

unsigned PresburgerSet::getNumFACs() const {
  return flatAffineConstraints.size();
}

unsigned PresburgerSet::getNumDims() const { return nDim; }

unsigned PresburgerSet::getNumSyms() const { return nSym; }

ArrayRef<FlatAffineConstraints>
PresburgerSet::getAllFlatAffineConstraints() const {
  return flatAffineConstraints;
}

const FlatAffineConstraints &
PresburgerSet::getFlatAffineConstraints(unsigned index) const {
  assert(index < flatAffineConstraints.size() && "index out of bounds!");
  return flatAffineConstraints[index];
}

/// Assert that the FlatAffineConstraints and PresburgerSet live in
/// compatible spaces.
static void assertDimensionsCompatible(const FlatAffineConstraints &fac,
                                       const PresburgerSet &set) {
  assert(fac.getNumDimIds() == set.getNumDims() &&
         "Number of dimensions of the FlatAffineConstraints and PresburgerSet"
         "do not match!");
  assert(fac.getNumSymbolIds() == set.getNumSyms() &&
         "Number of symbols of the FlatAffineConstraints and PresburgerSet"
         "do not match!");
}

/// Assert that the two PresburgerSets live in compatible spaces.
static void assertDimensionsCompatible(const PresburgerSet &setA,
                                       const PresburgerSet &setB) {
  assert(setA.getNumDims() == setB.getNumDims() &&
         "Number of dimensions of the PresburgerSets do not match!");
  assert(setA.getNumSyms() == setB.getNumSyms() &&
         "Number of symbols of the PresburgerSets do not match!");
}

/// Mutate this set, turning it into the union of this set and the given
/// FlatAffineConstraints.
void PresburgerSet::unionFACInPlace(const FlatAffineConstraints &fac) {
  assertDimensionsCompatible(fac, *this);
  flatAffineConstraints.push_back(fac);
}

/// Mutate this set, turning it into the union of this set and the given set.
///
/// This is accomplished by simply adding all the FACs of the given set to this
/// set.
void PresburgerSet::unionSetInPlace(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);
  for (const FlatAffineConstraints &fac : set.flatAffineConstraints)
    unionFACInPlace(fac);
}

/// Return the union of this set and the given set.
PresburgerSet PresburgerSet::unionSet(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);
  PresburgerSet result = *this;
  result.unionSetInPlace(set);
  return result;
}

/// A point is contained in the union iff any of the parts contain the point.
bool PresburgerSet::containsPoint(ArrayRef<int64_t> point) const {
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    if (fac.containsPoint(point))
      return true;
  }
  return false;
}

PresburgerSet PresburgerSet::getUniverse(unsigned nDim, unsigned nSym) {
  PresburgerSet result(nDim, nSym);
  result.unionFACInPlace(FlatAffineConstraints::getUniverse(nDim, nSym));
  return result;
}

PresburgerSet PresburgerSet::getEmptySet(unsigned nDim, unsigned nSym) {
  return PresburgerSet(nDim, nSym);
}

// Return the intersection of this set with the given set.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
PresburgerSet PresburgerSet::intersect(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);

  PresburgerSet result(nDim, nSym);
  for (const FlatAffineConstraints &csA : flatAffineConstraints) {
    for (const FlatAffineConstraints &csB : set.flatAffineConstraints) {
      FlatAffineConstraints intersection(csA);
      intersection.append(csB);
      if (!intersection.isEmpty())
        result.unionFACInPlace(std::move(intersection));
    }
  }
  return result;
}

/// Return `coeffs` with all the elements negated.
static SmallVector<int64_t, 8> getNegatedCoeffs(ArrayRef<int64_t> coeffs) {
  SmallVector<int64_t, 8> negatedCoeffs;
  negatedCoeffs.reserve(coeffs.size());
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  return negatedCoeffs;
}

/// Return the complement of the given inequality.
///
/// The complement of a_1 x_1 + ... + a_n x_ + c >= 0 is
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0.
static SmallVector<int64_t, 8> getComplementIneq(ArrayRef<int64_t> ineq) {
  SmallVector<int64_t, 8> coeffs;
  coeffs.reserve(ineq.size());
  for (int64_t coeff : ineq)
    coeffs.emplace_back(-coeff);
  --coeffs.back();
  return coeffs;
}

/// Return the set difference b \ s and accumulate the result into `result`.
/// `simplex` must correspond to b.
///
/// In the following, V denotes union, ^ denotes intersection, \ denotes set
/// difference and ~ denotes complement.
/// Let b be the FlatAffineConstraints and s = (V_i s_i) be the set. We want
/// b \ (V_i s_i).
///
/// Let s_i = ^_j s_ij, where each s_ij is a single inequality. To compute
/// b \ s_i = b ^ ~s_i, we partition s_i based on the first violated inequality:
/// ~s_i = (~s_i1) V (s_i1 ^ ~s_i2) V (s_i1 ^ s_i2 ^ ~s_i3) V ...
/// And the required result is (b ^ ~s_i1) V (b ^ s_i1 ^ ~s_i2) V ...
/// We recurse by subtracting V_{j > i} S_j from each of these parts and
/// returning the union of the results. Each equality is handled as a
/// conjunction of two inequalities.
///
/// As a heuristic, we try adding all the constraints and check if simplex
/// says that the intersection is empty. Also, in the process we find out that
/// some constraints are redundant. These redundant constraints are ignored.
static void subtractRecursively(FlatAffineConstraints &b, Simplex &simplex,
                                const PresburgerSet &s, unsigned i,
                                PresburgerSet &result) {
  if (i == s.getNumFACs()) {
    result.unionFACInPlace(b);
    return;
  }
  const FlatAffineConstraints &sI = s.getFlatAffineConstraints(i);
  unsigned initialSnapshot = simplex.getSnapshot();
  unsigned offset = simplex.numConstraints();
  simplex.intersectFlatAffineConstraints(sI);

  if (simplex.isEmpty()) {
    /// b ^ s_i is empty, so b \ s_i = b. We move directly to i + 1.
    simplex.rollback(initialSnapshot);
    subtractRecursively(b, simplex, s, i + 1, result);
    return;
  }

  simplex.detectRedundant();
  llvm::SmallBitVector isMarkedRedundant;
  for (unsigned j = 0; j < 2 * sI.getNumEqualities() + sI.getNumInequalities();
       j++)
    isMarkedRedundant.push_back(simplex.isMarkedRedundant(offset + j));

  simplex.rollback(initialSnapshot);

  // Recurse with the part b ^ ~ineq. Note that b is modified throughout
  // subtractRecursively. At the time this function is called, the current b is
  // actually equal to b ^ s_i1 ^ s_i2 ^ ... ^ s_ij, and ineq is the next
  // inequality, s_{i,j+1}. This function recurses into the next level i + 1
  // with the part b ^ s_i1 ^ s_i2 ^ ... ^ s_ij ^ ~s_{i,j+1}.
  auto recurseWithInequality = [&, i](ArrayRef<int64_t> ineq) {
    size_t snapshot = simplex.getSnapshot();
    b.addInequality(ineq);
    simplex.addInequality(ineq);
    subtractRecursively(b, simplex, s, i + 1, result);
    b.removeInequality(b.getNumInequalities() - 1);
    simplex.rollback(snapshot);
  };

  // For each inequality ineq, we first recurse with the part where ineq
  // is not satisfied, and then add the ineq to b and simplex because
  // ineq must be satisfied by all later parts.
  auto processInequality = [&](ArrayRef<int64_t> ineq) {
    recurseWithInequality(getComplementIneq(ineq));
    b.addInequality(ineq);
    simplex.addInequality(ineq);
  };

  // processInequality appends some additional constraints to b. We want to
  // rollback b to its initial state before returning, which we will do by
  // removing all constraints beyond the original number of inequalities
  // and equalities, so we store these counts first.
  unsigned originalNumIneqs = b.getNumInequalities();
  unsigned originalNumEqs = b.getNumEqualities();

  for (unsigned j = 0, e = sI.getNumInequalities(); j < e; j++) {
    if (isMarkedRedundant[j])
      continue;
    processInequality(sI.getInequality(j));
  }

  offset = sI.getNumInequalities();
  for (unsigned j = 0, e = sI.getNumEqualities(); j < e; ++j) {
    const ArrayRef<int64_t> &coeffs = sI.getEquality(j);
    // Same as the above loop for inequalities, done once each for the positive
    // and negative inequalities that make up this equality.
    if (!isMarkedRedundant[offset + 2 * j])
      processInequality(coeffs);
    if (!isMarkedRedundant[offset + 2 * j + 1])
      processInequality(getNegatedCoeffs(coeffs));
  }

  // Rollback b and simplex to their initial states.
  for (unsigned i = b.getNumInequalities(); i > originalNumIneqs; --i)
    b.removeInequality(i - 1);

  for (unsigned i = b.getNumEqualities(); i > originalNumEqs; --i)
    b.removeEquality(i - 1);

  simplex.rollback(initialSnapshot);
}

/// Return the set difference fac \ set.
///
/// The FAC here is modified in subtractRecursively, so it cannot be a const
/// reference even though it is restored to its original state before returning
/// from that function.
PresburgerSet PresburgerSet::getSetDifference(FlatAffineConstraints fac,
                                              const PresburgerSet &set) {
  assertDimensionsCompatible(fac, set);
  if (fac.isEmptyByGCDTest())
    return PresburgerSet::getEmptySet(fac.getNumDimIds(),
                                      fac.getNumSymbolIds());

  PresburgerSet result(fac.getNumDimIds(), fac.getNumSymbolIds());
  Simplex simplex(fac);
  subtractRecursively(fac, simplex, set, 0, result);
  return result;
}

/// Return the complement of this set.
PresburgerSet PresburgerSet::complement() const {
  return getSetDifference(
      FlatAffineConstraints::getUniverse(getNumDims(), getNumSyms()), *this);
}

/// Return the result of subtract the given set from this set, i.e.,
/// return `this \ set`.
PresburgerSet PresburgerSet::subtract(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);
  PresburgerSet result(nDim, nSym);
  // We compute (V_i t_i) \ (V_i set_i) as V_i (t_i \ V_i set_i).
  for (const FlatAffineConstraints &fac : flatAffineConstraints)
    result.unionSetInPlace(getSetDifference(fac, set));
  return result;
}

/// Return true if all the sets in the union are known to be integer empty,
/// false otherwise.
bool PresburgerSet::isIntegerEmpty() const {
  assert(nSym == 0 && "isIntegerEmpty is intended for non-symbolic sets");
  // The set is empty iff all of the disjuncts are empty.
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    if (!fac.isIntegerEmpty())
      return false;
  }
  return true;
}

bool PresburgerSet::findIntegerSample(SmallVectorImpl<int64_t> &sample) {
  assert(nSym == 0 && "findIntegerSample is intended for non-symbolic sets");
  // A sample exists iff any of the disjuncts contains a sample.
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    if (Optional<SmallVector<int64_t, 8>> opt = fac.findIntegerSample()) {
      sample = std::move(*opt);
      return true;
    }
  }
  return false;
}

void PresburgerSet::print(raw_ostream &os) const {
  os << getNumFACs() << " FlatAffineConstraints:\n";
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    fac.print(os);
    os << '\n';
  }
}

void PresburgerSet::dump() const { print(llvm::errs()); }
