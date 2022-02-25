//===- Set.cpp - MLIR PresburgerSet Class ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSet.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace presburger_utils;

PresburgerSet::PresburgerSet(const IntegerPolyhedron &poly)
    : PresburgerSpace(poly) {
  unionPolyInPlace(poly);
}

unsigned PresburgerSet::getNumPolys() const {
  return integerPolyhedrons.size();
}

ArrayRef<IntegerPolyhedron> PresburgerSet::getAllIntegerPolyhedron() const {
  return integerPolyhedrons;
}

const IntegerPolyhedron &
PresburgerSet::getIntegerPolyhedron(unsigned index) const {
  assert(index < integerPolyhedrons.size() && "index out of bounds!");
  return integerPolyhedrons[index];
}

/// Assert that the IntegerPolyhedron and PresburgerSet live in
/// compatible spaces.
static void assertDimensionsCompatible(const IntegerPolyhedron &poly,
                                       const PresburgerSet &set) {
  assert(poly.getNumDimIds() == set.getNumDimIds() &&
         "Number of dimensions of the IntegerPolyhedron and PresburgerSet"
         "do not match!");
  assert(poly.getNumSymbolIds() == set.getNumSymbolIds() &&
         "Number of symbols of the IntegerPolyhedron and PresburgerSet"
         "do not match!");
}

/// Assert that the two PresburgerSets live in compatible spaces.
static void assertDimensionsCompatible(const PresburgerSet &setA,
                                       const PresburgerSet &setB) {
  assert(setA.getNumDimIds() == setB.getNumDimIds() &&
         "Number of dimensions of the PresburgerSets do not match!");
  assert(setA.getNumSymbolIds() == setB.getNumSymbolIds() &&
         "Number of symbols of the PresburgerSets do not match!");
}

/// Mutate this set, turning it into the union of this set and the given
/// IntegerPolyhedron.
void PresburgerSet::unionPolyInPlace(const IntegerPolyhedron &poly) {
  assertDimensionsCompatible(poly, *this);
  integerPolyhedrons.push_back(poly);
}

/// Mutate this set, turning it into the union of this set and the given set.
///
/// This is accomplished by simply adding all the Poly of the given set to this
/// set.
void PresburgerSet::unionSetInPlace(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);
  for (const IntegerPolyhedron &poly : set.integerPolyhedrons)
    unionPolyInPlace(poly);
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
  return llvm::any_of(integerPolyhedrons, [&](const IntegerPolyhedron &poly) {
    return (poly.containsPoint(point));
  });
}

PresburgerSet PresburgerSet::getUniverse(unsigned numDims,
                                         unsigned numSymbols) {
  PresburgerSet result(numDims, numSymbols);
  result.unionPolyInPlace(IntegerPolyhedron::getUniverse(numDims, numSymbols));
  return result;
}

PresburgerSet PresburgerSet::getEmptySet(unsigned numDims,
                                         unsigned numSymbols) {
  return PresburgerSet(numDims, numSymbols);
}

// Return the intersection of this set with the given set.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
//
// If S_i or T_j have local variables, then S_i and T_j contains the local
// variables of both.
PresburgerSet PresburgerSet::intersect(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);

  PresburgerSet result(getNumDimIds(), getNumSymbolIds());
  for (const IntegerPolyhedron &csA : integerPolyhedrons) {
    for (const IntegerPolyhedron &csB : set.integerPolyhedrons) {
      IntegerPolyhedron csACopy = csA, csBCopy = csB;
      csACopy.mergeLocalIds(csBCopy);
      csACopy.append(csBCopy);
      if (!csACopy.isEmpty())
        result.unionPolyInPlace(csACopy);
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
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0,
/// since all the variables are constrained to be integers.
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
/// In the following, U denotes union, ^ denotes intersection, \ denotes set
/// difference and ~ denotes complement.
/// Let b be the IntegerPolyhedron and s = (U_i s_i) be the set. We want
/// b \ (U_i s_i).
///
/// Let s_i = ^_j s_ij, where each s_ij is a single inequality. To compute
/// b \ s_i = b ^ ~s_i, we partition s_i based on the first violated inequality:
/// ~s_i = (~s_i1) U (s_i1 ^ ~s_i2) U (s_i1 ^ s_i2 ^ ~s_i3) U ...
/// And the required result is (b ^ ~s_i1) U (b ^ s_i1 ^ ~s_i2) U ...
/// We recurse by subtracting U_{j > i} S_j from each of these parts and
/// returning the union of the results. Each equality is handled as a
/// conjunction of two inequalities.
///
/// Note that the same approach works even if an inequality involves a floor
/// division. For example, the complement of x <= 7*floor(x/7) is still
/// x > 7*floor(x/7). Since b \ s_i contains the inequalities of both b and s_i
/// (or the complements of those inequalities), b \ s_i may contain the
/// divisions present in both b and s_i. Therefore, we need to add the local
/// division variables of both b and s_i to each part in the result. This means
/// adding the local variables of both b and s_i, as well as the corresponding
/// division inequalities to each part. Since the division inequalities are
/// added to each part, we can skip the parts where the complement of any
/// division inequality is added, as these parts will become empty anyway.
///
/// As a heuristic, we try adding all the constraints and check if simplex
/// says that the intersection is empty. If it is, then subtracting this Poly is
/// a no-op and we just skip it. Also, in the process we find out that some
/// constraints are redundant. These redundant constraints are ignored.
///
/// b and simplex are callee saved, i.e., their values on return are
/// semantically equivalent to their values when the function is called.
static void subtractRecursively(IntegerPolyhedron &b, Simplex &simplex,
                                const PresburgerSet &s, unsigned i,
                                PresburgerSet &result) {
  if (i == s.getNumPolys()) {
    result.unionPolyInPlace(b);
    return;
  }
  IntegerPolyhedron sI = s.getIntegerPolyhedron(i);

  // Below, we append some additional constraints and ids to b. We want to
  // rollback b to its initial state before returning, which we will do by
  // removing all constraints beyond the original number of inequalities
  // and equalities, so we store these counts first.
  const unsigned bInitNumIneqs = b.getNumInequalities();
  const unsigned bInitNumEqs = b.getNumEqualities();
  const unsigned bInitNumLocals = b.getNumLocalIds();
  // Similarly, we also want to rollback simplex to its original state.
  const unsigned initialSnapshot = simplex.getSnapshot();

  auto restoreState = [&]() {
    b.removeIdRange(IntegerPolyhedron::IdKind::Local, bInitNumLocals,
                    b.getNumLocalIds());
    b.removeInequalityRange(bInitNumIneqs, b.getNumInequalities());
    b.removeEqualityRange(bInitNumEqs, b.getNumEqualities());
    simplex.rollback(initialSnapshot);
  };

  // Automatically restore the original state when we return.
  auto stateRestorer = llvm::make_scope_exit(restoreState);

  // Find out which inequalities of sI correspond to division inequalities for
  // the local variables of sI.
  std::vector<MaybeLocalRepr> repr(sI.getNumLocalIds());
  sI.getLocalReprs(repr);

  // Add sI's locals to b, after b's locals. Also add b's locals to sI, before
  // sI's locals.
  b.mergeLocalIds(sI);

  // Mark which inequalities of sI are division inequalities and add all such
  // inequalities to b.
  llvm::SmallBitVector isDivInequality(sI.getNumInequalities());
  for (MaybeLocalRepr &maybeInequality : repr) {
    assert(maybeInequality.kind == ReprKind::Inequality &&
           "Subtraction is not supported when a representation of the local "
           "variables of the subtrahend cannot be found!");
    auto lb = maybeInequality.repr.inequalityPair.lowerBoundIdx;
    auto ub = maybeInequality.repr.inequalityPair.upperBoundIdx;

    b.addInequality(sI.getInequality(lb));
    b.addInequality(sI.getInequality(ub));

    assert(lb != ub &&
           "Upper and lower bounds must be different inequalities!");
    isDivInequality[lb] = true;
    isDivInequality[ub] = true;
  }

  unsigned offset = simplex.getNumConstraints();
  unsigned numLocalsAdded = b.getNumLocalIds() - bInitNumLocals;
  simplex.appendVariable(numLocalsAdded);

  unsigned snapshotBeforeIntersect = simplex.getSnapshot();
  simplex.intersectIntegerPolyhedron(sI);

  if (simplex.isEmpty()) {
    // b ^ s_i is empty, so b \ s_i = b. We move directly to i + 1.
    // We are ignoring level i completely, so we restore the state
    // *before* going to level i + 1.
    restoreState();
    subtractRecursively(b, simplex, s, i + 1, result);

    // We already restored the state above and the recursive call should have
    // restored to the same state before returning, so we don't need to restore
    // the state again.
    stateRestorer.release();
    return;
  }

  simplex.detectRedundant();

  // Equalities are added to simplex as a pair of inequalities.
  unsigned totalNewSimplexInequalities =
      2 * sI.getNumEqualities() + sI.getNumInequalities();
  llvm::SmallBitVector isMarkedRedundant(totalNewSimplexInequalities);
  for (unsigned j = 0; j < totalNewSimplexInequalities; j++)
    isMarkedRedundant[j] = simplex.isMarkedRedundant(offset + j);

  simplex.rollback(snapshotBeforeIntersect);

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

  // Process all the inequalities, ignoring redundant inequalities and division
  // inequalities. The result is correct whether or not we ignore these, but
  // ignoring them makes the result simpler.
  for (unsigned j = 0, e = sI.getNumInequalities(); j < e; j++) {
    if (isMarkedRedundant[j])
      continue;
    if (isDivInequality[j])
      continue;
    processInequality(sI.getInequality(j));
  }

  offset = sI.getNumInequalities();
  for (unsigned j = 0, e = sI.getNumEqualities(); j < e; ++j) {
    ArrayRef<int64_t> coeffs = sI.getEquality(j);
    // For each equality, process the positive and negative inequalities that
    // make up this equality. If Simplex found an inequality to be redundant, we
    // skip it as above to make the result simpler. Divisions are always
    // represented in terms of inequalities and not equalities, so we do not
    // check for division inequalities here.
    if (!isMarkedRedundant[offset + 2 * j])
      processInequality(coeffs);
    if (!isMarkedRedundant[offset + 2 * j + 1])
      processInequality(getNegatedCoeffs(coeffs));
  }
}

/// Return the set difference poly \ set.
///
/// The Poly here is modified in subtractRecursively, so it cannot be a const
/// reference even though it is restored to its original state before returning
/// from that function.
PresburgerSet PresburgerSet::getSetDifference(IntegerPolyhedron poly,
                                              const PresburgerSet &set) {
  assertDimensionsCompatible(poly, set);
  if (poly.isEmptyByGCDTest())
    return PresburgerSet::getEmptySet(poly.getNumDimIds(),
                                      poly.getNumSymbolIds());

  PresburgerSet result(poly.getNumDimIds(), poly.getNumSymbolIds());
  Simplex simplex(poly);
  subtractRecursively(poly, simplex, set, 0, result);
  return result;
}

/// Return the complement of this set.
PresburgerSet PresburgerSet::complement() const {
  return getSetDifference(
      IntegerPolyhedron::getUniverse(getNumDimIds(), getNumSymbolIds()), *this);
}

/// Return the result of subtract the given set from this set, i.e.,
/// return `this \ set`.
PresburgerSet PresburgerSet::subtract(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);
  PresburgerSet result(getNumDimIds(), getNumSymbolIds());
  // We compute (U_i t_i) \ (U_i set_i) as U_i (t_i \ V_i set_i).
  for (const IntegerPolyhedron &poly : integerPolyhedrons)
    result.unionSetInPlace(getSetDifference(poly, set));
  return result;
}

/// T is a subset of S iff T \ S is empty, since if T \ S contains a
/// point then this is a point that is contained in T but not S, and
/// if T contains a point that is not in S, this also lies in T \ S.
bool PresburgerSet::isSubsetOf(const PresburgerSet &set) const {
  return this->subtract(set).isIntegerEmpty();
}

/// Two sets are equal iff they are subsets of each other.
bool PresburgerSet::isEqual(const PresburgerSet &set) const {
  assertDimensionsCompatible(set, *this);
  return this->isSubsetOf(set) && set.isSubsetOf(*this);
}

/// Return true if all the sets in the union are known to be integer empty,
/// false otherwise.
bool PresburgerSet::isIntegerEmpty() const {
  // The set is empty iff all of the disjuncts are empty.
  return std::all_of(
      integerPolyhedrons.begin(), integerPolyhedrons.end(),
      [](const IntegerPolyhedron &poly) { return poly.isIntegerEmpty(); });
}

bool PresburgerSet::findIntegerSample(SmallVectorImpl<int64_t> &sample) {
  // A sample exists iff any of the disjuncts contains a sample.
  for (const IntegerPolyhedron &poly : integerPolyhedrons) {
    if (Optional<SmallVector<int64_t, 8>> opt = poly.findIntegerSample()) {
      sample = std::move(*opt);
      return true;
    }
  }
  return false;
}

Optional<uint64_t> PresburgerSet::computeVolume() const {
  assert(getNumSymbolIds() == 0 && "Symbols are not yet supported!");
  // The sum of the volumes of the disjuncts is a valid overapproximation of the
  // volume of their union, even if they overlap.
  uint64_t result = 0;
  for (const IntegerPolyhedron &poly : integerPolyhedrons) {
    Optional<uint64_t> volume = poly.computeVolume();
    if (!volume)
      return {};
    result += *volume;
  }
  return result;
}

PresburgerSet PresburgerSet::coalesce() const {
  PresburgerSet newSet =
      PresburgerSet::getEmptySet(getNumDimIds(), getNumSymbolIds());
  llvm::SmallBitVector isRedundant(getNumPolys());

  for (unsigned i = 0, e = integerPolyhedrons.size(); i < e; ++i) {
    if (isRedundant[i])
      continue;
    Simplex simplex(integerPolyhedrons[i]);

    // Check whether the polytope of `simplex` is empty. If so, it is trivially
    // redundant.
    if (simplex.isEmpty()) {
      isRedundant[i] = true;
      continue;
    }

    // Check whether `IntegerPolyhedron[i]` is contained in any Poly, that is
    // different from itself and not yet marked as redundant.
    for (unsigned j = 0, e = integerPolyhedrons.size(); j < e; ++j) {
      if (j == i || isRedundant[j])
        continue;

      if (simplex.isRationalSubsetOf(integerPolyhedrons[j])) {
        isRedundant[i] = true;
        break;
      }
    }
  }

  for (unsigned i = 0, e = integerPolyhedrons.size(); i < e; ++i)
    if (!isRedundant[i])
      newSet.unionPolyInPlace(integerPolyhedrons[i]);

  return newSet;
}

void PresburgerSet::print(raw_ostream &os) const {
  os << getNumPolys() << " IntegerPolyhedron:\n";
  for (const IntegerPolyhedron &poly : integerPolyhedrons) {
    poly.print(os);
    os << '\n';
  }
}

void PresburgerSet::dump() const { print(llvm::errs()); }
