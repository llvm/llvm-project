//===- PWMAFunction.cpp - MLIR PWMAFunction Class -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Simplex.h"

using namespace mlir;
using namespace presburger;

// Return the result of subtracting the two given vectors pointwise.
// The vectors must be of the same size.
// e.g., [3, 4, 6] - [2, 5, 1] = [1, -1, 5].
static SmallVector<int64_t, 8> subtract(ArrayRef<int64_t> vecA,
                                        ArrayRef<int64_t> vecB) {
  assert(vecA.size() == vecB.size() &&
         "Cannot subtract vectors of differing lengths!");
  SmallVector<int64_t, 8> result;
  result.reserve(vecA.size());
  for (unsigned i = 0, e = vecA.size(); i < e; ++i)
    result.push_back(vecA[i] - vecB[i]);
  return result;
}

PresburgerSet PWMAFunction::getDomain() const {
  PresburgerSet domain = PresburgerSet::getEmpty(getSpace());
  for (const MultiAffineFunction &piece : pieces)
    domain.unionInPlace(piece.getDomain());
  return domain;
}

Optional<SmallVector<int64_t, 8>>
MultiAffineFunction::valueAt(ArrayRef<int64_t> point) const {
  assert(point.size() == domainSet.getNumDimAndSymbolVars() &&
         "Point has incorrect dimensionality!");

  Optional<SmallVector<int64_t, 8>> maybeLocalValues =
      getDomain().containsPointNoLocal(point);
  if (!maybeLocalValues)
    return {};

  // The point lies in the domain, so we need to compute the output value.
  SmallVector<int64_t, 8> pointHomogenous{llvm::to_vector(point)};
  // The given point didn't include the values of locals which the output is a
  // function of; we have computed one possible set of values and use them
  // here. The function is not allowed to have local vars that take more than
  // one possible value.
  pointHomogenous.append(*maybeLocalValues);
  // The matrix `output` has an affine expression in the ith row, corresponding
  // to the expression for the ith value in the output vector. The last column
  // of the matrix contains the constant term. Let v be the input point with
  // a 1 appended at the end. We can see that output * v gives the desired
  // output vector.
  pointHomogenous.emplace_back(1);
  SmallVector<int64_t, 8> result =
      output.postMultiplyWithColumn(pointHomogenous);
  assert(result.size() == getNumOutputs());
  return result;
}

Optional<SmallVector<int64_t, 8>>
PWMAFunction::valueAt(ArrayRef<int64_t> point) const {
  assert(point.size() == getNumInputs() &&
         "Point has incorrect dimensionality!");
  for (const MultiAffineFunction &piece : pieces)
    if (Optional<SmallVector<int64_t, 8>> output = piece.valueAt(point))
      return output;
  return {};
}

void MultiAffineFunction::print(raw_ostream &os) const {
  os << "Domain:";
  domainSet.print(os);
  os << "Output:\n";
  output.print(os);
  os << "\n";
}

void MultiAffineFunction::dump() const { print(llvm::errs()); }

bool MultiAffineFunction::isEqual(const MultiAffineFunction &other) const {
  return getDomainSpace().isCompatible(other.getDomainSpace()) &&
         getDomain().isEqual(other.getDomain()) &&
         isEqualWhereDomainsOverlap(other);
}

unsigned MultiAffineFunction::insertVar(VarKind kind, unsigned pos,
                                        unsigned num) {
  assert(kind != VarKind::Domain && "Domain has to be zero in a set");
  unsigned absolutePos = domainSet.getVarKindOffset(kind) + pos;
  output.insertColumns(absolutePos, num);
  return domainSet.insertVar(kind, pos, num);
}

void MultiAffineFunction::removeVarRange(VarKind kind, unsigned varStart,
                                         unsigned varLimit) {
  output.removeColumns(varStart + domainSet.getVarKindOffset(kind),
                       varLimit - varStart);
  domainSet.removeVarRange(kind, varStart, varLimit);
}

void MultiAffineFunction::truncateOutput(unsigned count) {
  assert(count <= output.getNumRows());
  output.resizeVertically(count);
}

void PWMAFunction::truncateOutput(unsigned count) {
  assert(count <= numOutputs);
  for (MultiAffineFunction &piece : pieces)
    piece.truncateOutput(count);
  numOutputs = count;
}

void MultiAffineFunction::mergeLocalVars(MultiAffineFunction &other) {
  // Merge output local vars of both functions without using division
  // information i.e. append local vars of `other` to `this` and insert
  // local vars of `this` to `other` at the start of it's local vars.
  output.insertColumns(domainSet.getVarKindEnd(VarKind::Local),
                       other.domainSet.getNumLocalVars());
  other.output.insertColumns(other.domainSet.getVarKindOffset(VarKind::Local),
                             domainSet.getNumLocalVars());

  auto merge = [this, &other](unsigned i, unsigned j) -> bool {
    // Merge local at position j into local at position i in function domain.
    domainSet.eliminateRedundantLocalVar(i, j);
    other.domainSet.eliminateRedundantLocalVar(i, j);

    unsigned localOffset = domainSet.getVarKindOffset(VarKind::Local);

    // Merge local at position j into local at position i in output domain.
    output.addToColumn(localOffset + j, localOffset + i, 1);
    output.removeColumn(localOffset + j);
    other.output.addToColumn(localOffset + j, localOffset + i, 1);
    other.output.removeColumn(localOffset + j);

    return true;
  };

  presburger::mergeLocalVars(domainSet, other.domainSet, merge);
}

bool MultiAffineFunction::isEqualWhereDomainsOverlap(
    MultiAffineFunction other) const {
  if (!getDomainSpace().isCompatible(other.getDomainSpace()))
    return false;

  // `commonFunc` has the same output as `this`.
  MultiAffineFunction commonFunc = *this;
  // After this merge, `commonFunc` and `other` have the same local vars; they
  // are merged.
  commonFunc.mergeLocalVars(other);
  // After this, the domain of `commonFunc` will be the intersection of the
  // domains of `this` and `other`.
  commonFunc.domainSet.append(other.domainSet);

  // `commonDomainMatching` contains the subset of the common domain
  // where the outputs of `this` and `other` match.
  //
  // We want to add constraints equating the outputs of `this` and `other`.
  // However, `this` may have difference local vars from `other`, whereas we
  // need both to have the same locals. Accordingly, we use `commonFunc.output`
  // in place of `this->output`, since `commonFunc` has the same output but also
  // has its locals merged.
  IntegerPolyhedron commonDomainMatching = commonFunc.getDomain();
  for (unsigned row = 0, e = getNumOutputs(); row < e; ++row)
    commonDomainMatching.addEquality(
        subtract(commonFunc.output.getRow(row), other.output.getRow(row)));

  // If the whole common domain is a subset of commonDomainMatching, then they
  // are equal and the two functions match on the whole common domain.
  return commonFunc.getDomain().isSubsetOf(commonDomainMatching);
}

/// Two PWMAFunctions are equal if they have the same dimensionalities,
/// the same domain, and take the same value at every point in the domain.
bool PWMAFunction::isEqual(const PWMAFunction &other) const {
  if (!space.isCompatible(other.space))
    return false;

  if (!this->getDomain().isEqual(other.getDomain()))
    return false;

  // Check if, whenever the domains of a piece of `this` and a piece of `other`
  // overlap, they take the same output value. If `this` and `other` have the
  // same domain (checked above), then this check passes iff the two functions
  // have the same output at every point in the domain.
  for (const MultiAffineFunction &aPiece : this->pieces)
    for (const MultiAffineFunction &bPiece : other.pieces)
      if (!aPiece.isEqualWhereDomainsOverlap(bPiece))
        return false;
  return true;
}

void PWMAFunction::addPiece(const MultiAffineFunction &piece) {
  assert(space.isCompatible(piece.getDomainSpace()) &&
         "Piece to be added is not compatible with this PWMAFunction!");
  assert(piece.isConsistent() && "Piece is internally inconsistent!");
  assert(this->getDomain()
             .intersect(PresburgerSet(piece.getDomain()))
             .isIntegerEmpty() &&
         "New piece's domain overlaps with that of existing pieces!");
  pieces.push_back(piece);
}

void PWMAFunction::addPiece(const IntegerPolyhedron &domain,
                            const Matrix &output) {
  addPiece(MultiAffineFunction(domain, output));
}

void PWMAFunction::addPiece(const PresburgerSet &domain, const Matrix &output) {
  for (const IntegerRelation &newDom : domain.getAllDisjuncts())
    addPiece(IntegerPolyhedron(newDom), output);
}

void PWMAFunction::print(raw_ostream &os) const {
  os << pieces.size() << " pieces:\n";
  for (const MultiAffineFunction &piece : pieces)
    piece.print(os);
}

void PWMAFunction::dump() const { print(llvm::errs()); }

PWMAFunction PWMAFunction::unionFunction(
    const PWMAFunction &func,
    llvm::function_ref<PresburgerSet(MultiAffineFunction maf1,
                                     MultiAffineFunction maf2)>
        tiebreak) const {
  assert(getNumOutputs() == func.getNumOutputs() &&
         "Number of outputs of functions should be same.");
  assert(getSpace().isCompatible(func.getSpace()) &&
         "Space is not compatible.");

  // The algorithm used here is as follows:
  // - Add the output of funcB for the part of the domain where both funcA and
  //   funcB are defined, and `tiebreak` chooses the output of funcB.
  // - Add the output of funcA, where funcB is not defined or `tiebreak` chooses
  //   funcA over funcB.
  // - Add the output of funcB, where funcA is not defined.

  // Add parts of the common domain where funcB's output is used. Also
  // add all the parts where funcA's output is used, both common and non-common.
  PWMAFunction result(getSpace(), getNumOutputs());
  for (const MultiAffineFunction &funcA : pieces) {
    PresburgerSet dom(funcA.getDomain());
    for (const MultiAffineFunction &funcB : func.pieces) {
      PresburgerSet better = tiebreak(funcB, funcA);
      // Add the output of funcB, where it is better than output of funcA.
      // The disjuncts in "better" will be disjoint as tiebreak should gurantee
      // that.
      result.addPiece(better, funcB.getOutputMatrix());
      dom = dom.subtract(better);
    }
    // Add output of funcA, where it is better than funcB, or funcB is not
    // defined.
    //
    // `dom` here is guranteed to be disjoint from already added pieces
    // because because the pieces added before are either:
    // - Subsets of the domain of other MAFs in `this`, which are guranteed
    //   to be disjoint from `dom`, or
    // - They are one of the pieces added for `funcB`, and we have been
    //   subtracting all such pieces from `dom`, so `dom` is disjoint from those
    //   pieces as well.
    result.addPiece(dom, funcA.getOutputMatrix());
  }

  // Add parts of funcB which are not shared with funcA.
  PresburgerSet dom = getDomain();
  for (const MultiAffineFunction &funcB : func.pieces)
    result.addPiece(funcB.getDomain().subtract(dom), funcB.getOutputMatrix());

  return result;
}

/// A tiebreak function which breaks ties by comparing the outputs
/// lexicographically. If `lexMin` is true, then the ties are broken by
/// taking the lexicographically smaller output and otherwise, by taking the
/// lexicographically larger output.
template <bool lexMin>
static PresburgerSet tiebreakLex(const MultiAffineFunction &mafA,
                                 const MultiAffineFunction &mafB) {
  // TODO: Support local variables here.
  assert(mafA.getDomainSpace().isCompatible(mafB.getDomainSpace()) &&
         "Domain spaces should be compatible.");
  assert(mafA.getNumOutputs() == mafB.getNumOutputs() &&
         "Number of outputs of both functions should be same.");
  assert(mafA.getDomain().getNumLocalVars() == 0 &&
         "Local variables are not supported yet.");

  PresburgerSpace compatibleSpace = mafA.getDomain().getSpaceWithoutLocals();
  const PresburgerSpace &space = mafA.getDomain().getSpace();

  // We first create the set `result`, corresponding to the set where output
  // of mafA is lexicographically larger/smaller than mafB. This is done by
  // creating a PresburgerSet with the following constraints:
  //
  //    (outA[0] > outB[0]) U
  //    (outA[0] = outB[0], outA[1] > outA[1]) U
  //    (outA[0] = outB[0], outA[1] = outA[1], outA[2] > outA[2]) U
  //    ...
  //    (outA[0] = outB[0], ..., outA[n-2] = outB[n-2], outA[n-1] > outB[n-1])
  //
  // where `n` is the number of outputs.
  // If `lexMin` is set, the complement inequality is used:
  //
  //    (outA[0] < outB[0]) U
  //    (outA[0] = outB[0], outA[1] < outA[1]) U
  //    (outA[0] = outB[0], outA[1] = outA[1], outA[2] < outA[2]) U
  //    ...
  //    (outA[0] = outB[0], ..., outA[n-2] = outB[n-2], outA[n-1] < outB[n-1])
  PresburgerSet result = PresburgerSet::getEmpty(compatibleSpace);
  IntegerPolyhedron levelSet(/*numReservedInequalities=*/1,
                             /*numReservedEqualities=*/mafA.getNumOutputs(),
                             /*numReservedCols=*/space.getNumVars() + 1, space);
  for (unsigned level = 0; level < mafA.getNumOutputs(); ++level) {

    // Create the expression `outA - outB` for this level.
    SmallVector<int64_t, 8> subExpr =
        subtract(mafA.getOutputExpr(level), mafB.getOutputExpr(level));

    if (lexMin) {
      // For lexMin, we add an upper bound of -1:
      //        outA - outB <= -1
      //        outA <= outB - 1
      //        outA < outB
      levelSet.addBound(IntegerPolyhedron::BoundType::UB, subExpr, -1);
    } else {
      // For lexMax, we add a lower bound of 1:
      //        outA - outB >= 1
      //        outA > outB + 1
      //        outA > outB
      levelSet.addBound(IntegerPolyhedron::BoundType::LB, subExpr, 1);
    }

    // Union the set with the result.
    result.unionInPlace(levelSet);
    // There is only 1 inequality in `levelSet`, so the index is always 0.
    levelSet.removeInequality(0);
    // Add equality `outA - outB == 0` for this level for next iteration.
    levelSet.addEquality(subExpr);
  }

  // We then intersect `result` with the domain of mafA and mafB, to only
  // tiebreak on the domain where both are defined.
  result = result.intersect(PresburgerSet(mafA.getDomain()))
               .intersect(PresburgerSet(mafB.getDomain()));

  return result;
}

PWMAFunction PWMAFunction::unionLexMin(const PWMAFunction &func) {
  return unionFunction(func, tiebreakLex</*lexMin=*/true>);
}

PWMAFunction PWMAFunction::unionLexMax(const PWMAFunction &func) {
  return unionFunction(func, tiebreakLex</*lexMin=*/false>);
}
