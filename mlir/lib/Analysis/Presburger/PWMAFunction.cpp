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
  assert(point.size() == getNumDimAndSymbolIds() &&
         "Point has incorrect dimensionality!");

  Optional<SmallVector<int64_t, 8>> maybeLocalValues =
      getDomain().containsPointNoLocal(point);
  if (!maybeLocalValues)
    return {};

  // The point lies in the domain, so we need to compute the output value.
  SmallVector<int64_t, 8> pointHomogenous{llvm::to_vector(point)};
  // The given point didn't include the values of locals which the output is a
  // function of; we have computed one possible set of values and use them
  // here. The function is not allowed to have local ids that take more than
  // one possible value.
  pointHomogenous.append(*maybeLocalValues);
  // The matrix `output` has an affine expression in the ith row, corresponding
  // to the expression for the ith value in the output vector. The last column
  // of the matrix contains the constant term. Let v be the input point with
  // a 1 appended at the end. We can see that output * v gives the desired
  // output vector.
  pointHomogenous.push_back(1);
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
  IntegerPolyhedron::print(os);
  os << "Output:\n";
  output.print(os);
  os << "\n";
}

void MultiAffineFunction::dump() const { print(llvm::errs()); }

bool MultiAffineFunction::isEqual(const MultiAffineFunction &other) const {
  return isSpaceCompatible(other) && getDomain().isEqual(other.getDomain()) &&
         isEqualWhereDomainsOverlap(other);
}

unsigned MultiAffineFunction::insertId(IdKind kind, unsigned pos,
                                       unsigned num) {
  assert((kind != IdKind::Domain || num == 0) &&
         "Domain has to be zero in a set");
  unsigned absolutePos = getIdKindOffset(kind) + pos;
  output.insertColumns(absolutePos, num);
  return IntegerPolyhedron::insertId(kind, pos, num);
}

void MultiAffineFunction::swapId(unsigned posA, unsigned posB) {
  output.swapColumns(posA, posB);
  IntegerPolyhedron::swapId(posA, posB);
}

void MultiAffineFunction::removeIdRange(IdKind kind, unsigned idStart,
                                        unsigned idLimit) {
  output.removeColumns(idStart + getIdKindOffset(kind), idLimit - idStart);
  IntegerPolyhedron::removeIdRange(kind, idStart, idLimit);
}

void MultiAffineFunction::eliminateRedundantLocalId(unsigned posA,
                                                    unsigned posB) {
  unsigned localOffset = getIdKindOffset(IdKind::Local);
  output.addToColumn(localOffset + posB, localOffset + posA, /*scale=*/1);
  IntegerPolyhedron::eliminateRedundantLocalId(posA, posB);
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

bool MultiAffineFunction::isEqualWhereDomainsOverlap(
    MultiAffineFunction other) const {
  if (!isSpaceCompatible(other))
    return false;

  // `commonFunc` has the same output as `this`.
  MultiAffineFunction commonFunc = *this;
  // After this merge, `commonFunc` and `other` have the same local ids; they
  // are merged.
  commonFunc.mergeLocalIds(other);
  // After this, the domain of `commonFunc` will be the intersection of the
  // domains of `this` and `other`.
  commonFunc.IntegerPolyhedron::append(other);

  // `commonDomainMatching` contains the subset of the common domain
  // where the outputs of `this` and `other` match.
  //
  // We want to add constraints equating the outputs of `this` and `other`.
  // However, `this` may have difference local ids from `other`, whereas we
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
  if (!isSpaceCompatible(other))
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
  assert(piece.isSpaceCompatible(*this) &&
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

void PWMAFunction::print(raw_ostream &os) const {
  os << pieces.size() << " pieces:\n";
  for (const MultiAffineFunction &piece : pieces)
    piece.print(os);
}

void PWMAFunction::dump() const { print(llvm::errs()); }
