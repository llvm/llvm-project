//===- IntegerPolyhedron.cpp - MLIR IntegerPolyhedron Class ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an integer polyhedron.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"

using namespace mlir;

std::unique_ptr<IntegerPolyhedron> IntegerPolyhedron::clone() const {
  return std::make_unique<IntegerPolyhedron>(*this);
}

void IntegerPolyhedron::reset(unsigned numReservedInequalities,
                              unsigned numReservedEqualities,
                              unsigned newNumReservedCols, unsigned newNumDims,
                              unsigned newNumSymbols, unsigned newNumLocals) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  *this = IntegerPolyhedron(numReservedInequalities, numReservedEqualities,
                            newNumReservedCols, newNumDims, newNumSymbols,
                            newNumLocals);
}

void IntegerPolyhedron::reset(unsigned newNumDims, unsigned newNumSymbols,
                              unsigned newNumLocals) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals);
}

void IntegerPolyhedron::append(const IntegerPolyhedron &other) {
  assert(other.getNumCols() == getNumCols());
  assert(other.getNumDimIds() == getNumDimIds());
  assert(other.getNumSymbolIds() == getNumSymbolIds());

  inequalities.reserveRows(inequalities.getNumRows() +
                           other.getNumInequalities());
  equalities.reserveRows(equalities.getNumRows() + other.getNumEqualities());

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

unsigned IntegerPolyhedron::insertDimId(unsigned pos, unsigned num) {
  return insertId(IdKind::Dimension, pos, num);
}

unsigned IntegerPolyhedron::insertSymbolId(unsigned pos, unsigned num) {
  return insertId(IdKind::Symbol, pos, num);
}

unsigned IntegerPolyhedron::insertLocalId(unsigned pos, unsigned num) {
  return insertId(IdKind::Local, pos, num);
}

unsigned IntegerPolyhedron::insertId(IdKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumIdKind(kind));

  unsigned absolutePos = getIdKindOffset(kind) + pos;
  if (kind == IdKind::Dimension)
    numDims += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  numIds += num;

  inequalities.insertColumns(absolutePos, num);
  equalities.insertColumns(absolutePos, num);

  return absolutePos;
}

unsigned IntegerPolyhedron::appendDimId(unsigned num) {
  unsigned pos = getNumDimIds();
  insertId(IdKind::Dimension, pos, num);
  return pos;
}

unsigned IntegerPolyhedron::appendSymbolId(unsigned num) {
  unsigned pos = getNumSymbolIds();
  insertId(IdKind::Symbol, pos, num);
  return pos;
}

unsigned IntegerPolyhedron::appendLocalId(unsigned num) {
  unsigned pos = getNumLocalIds();
  insertId(IdKind::Local, pos, num);
  return pos;
}

void IntegerPolyhedron::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned row = equalities.appendExtraRow();
  for (unsigned i = 0, e = eq.size(); i < e; ++i)
    equalities(row, i) = eq[i];
}

void IntegerPolyhedron::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = inEq.size(); i < e; ++i)
    inequalities(row, i) = inEq[i];
}

void IntegerPolyhedron::removeId(IdKind kind, unsigned pos) {
  removeIdRange(kind, pos, pos + 1);
}

void IntegerPolyhedron::removeId(unsigned pos) { removeIdRange(pos, pos + 1); }

void IntegerPolyhedron::removeIdRange(IdKind kind, unsigned idStart,
                                      unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind));
  removeIdRange(getIdKindOffset(kind) + idStart,
                getIdKindOffset(kind) + idLimit);
}

void IntegerPolyhedron::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit < getNumCols() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < numIds && "invalid idStart position");

  // TODO: Make 'removeIdRange' a lambda called from here.
  // Remove eliminated identifiers from the constraints..
  equalities.removeColumns(idStart, idLimit - idStart);
  inequalities.removeColumns(idStart, idLimit - idStart);

  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  unsigned numLocalsEliminated = 0;
  unsigned numColsEliminated = idLimit - idStart;
  if (idStart < numDims) {
    numDimsEliminated = std::min(numDims, idLimit) - idStart;
  }
  // Check how many local id's were removed. Note that our identifier order is
  // [dims, symbols, locals]. Local id start at position numDims + numSymbols.
  if (idLimit > numDims + numSymbols) {
    numLocalsEliminated = std::min(
        idLimit - std::max(idStart, numDims + numSymbols), getNumLocalIds());
  }
  unsigned numSymbolsEliminated =
      numColsEliminated - numDimsEliminated - numLocalsEliminated;

  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;
}

void IntegerPolyhedron::removeEquality(unsigned pos) {
  equalities.removeRow(pos);
}

void IntegerPolyhedron::removeInequality(unsigned pos) {
  inequalities.removeRow(pos);
}

void IntegerPolyhedron::removeEqualityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  equalities.removeRows(start, end - start);
}

void IntegerPolyhedron::removeInequalityRange(unsigned start, unsigned end) {
  if (start >= end)
    return;
  inequalities.removeRows(start, end - start);
}

void IntegerPolyhedron::swapId(unsigned posA, unsigned posB) {
  assert(posA < getNumIds() && "invalid position A");
  assert(posB < getNumIds() && "invalid position B");

  if (posA == posB)
    return;

  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    std::swap(atIneq(r, posA), atIneq(r, posB));
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++)
    std::swap(atEq(r, posA), atEq(r, posB));
}

unsigned IntegerPolyhedron::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return 0;
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind expected to be Dimension, Symbol or Local!");
}

unsigned IntegerPolyhedron::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return getNumDimIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  if (kind == IdKind::Local)
    return getNumLocalIds();
  llvm_unreachable("IdKind expected to be Dimension, Symbol or Local!");
}

void IntegerPolyhedron::clearConstraints() {
  equalities.resizeVertically(0);
  inequalities.resizeVertically(0);
}

/// Gather all lower and upper bounds of the identifier at `pos`, and
/// optionally any equalities on it. In addition, the bounds are to be
/// independent of identifiers in position range [`offset`, `offset` + `num`).
void IntegerPolyhedron::getLowerAndUpperBoundIndices(
    unsigned pos, SmallVectorImpl<unsigned> *lbIndices,
    SmallVectorImpl<unsigned> *ubIndices, SmallVectorImpl<unsigned> *eqIndices,
    unsigned offset, unsigned num) const {
  assert(pos < getNumIds() && "invalid position");
  assert(offset + num < getNumCols() && "invalid range");

  // Checks for a constraint that has a non-zero coeff for the identifiers in
  // the position range [offset, offset + num) while ignoring `pos`.
  auto containsConstraintDependentOnRange = [&](unsigned r, bool isEq) {
    unsigned c, f;
    auto cst = isEq ? getEquality(r) : getInequality(r);
    for (c = offset, f = offset + num; c < f; ++c) {
      if (c == pos)
        continue;
      if (cst[c] != 0)
        break;
    }
    return c < f;
  };

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // The bounds are to be independent of [offset, offset + num) columns.
    if (containsConstraintDependentOnRange(r, /*isEq=*/false))
      continue;
    if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices->push_back(r);
    } else if (atIneq(r, pos) <= -1) {
      // Upper bound.
      ubIndices->push_back(r);
    }
  }

  // An equality is both a lower and upper bound. Record any equalities
  // involving the pos^th identifier.
  if (!eqIndices)
    return;

  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) == 0)
      continue;
    if (containsConstraintDependentOnRange(r, /*isEq=*/true))
      continue;
    eqIndices->push_back(r);
  }
}

bool IntegerPolyhedron::hasConsistentState() const {
  if (!inequalities.hasConsistentState())
    return false;
  if (!equalities.hasConsistentState())
    return false;

  // Catches errors where numDims, numSymbols, numIds aren't consistent.
  if (numDims > numIds || numSymbols > numIds || numDims + numSymbols > numIds)
    return false;

  return true;
}

void IntegerPolyhedron::setAndEliminate(unsigned pos,
                                        ArrayRef<int64_t> values) {
  if (values.empty())
    return;
  assert(pos + values.size() <= getNumIds() &&
         "invalid position or too many values");
  // Setting x_j = p in sum_i a_i x_i + c is equivalent to adding p*a_j to the
  // constant term and removing the id x_j. We do this for all the ids
  // pos, pos + 1, ... pos + values.size() - 1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
      atIneq(r, getNumCols() - 1) += atIneq(r, pos + i) * values[i];
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++)
    for (unsigned i = 0, numVals = values.size(); i < numVals; ++i)
      atEq(r, getNumCols() - 1) += atEq(r, pos + i) * values[i];
  removeIdRange(pos, pos + values.size());
}

void IntegerPolyhedron::clearAndCopyFrom(const IntegerPolyhedron &other) {
  *this = other;
}

void IntegerPolyhedron::printSpace(raw_ostream &os) const {
  os << "\nConstraints (" << getNumDimIds() << " dims, " << getNumSymbolIds()
     << " symbols, " << getNumLocalIds() << " locals), (" << getNumConstraints()
     << " constraints)\n";
}

void IntegerPolyhedron::print(raw_ostream &os) const {
  assert(hasConsistentState());
  printSpace(os);
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atEq(i, j) << " ";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atIneq(i, j) << " ";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void IntegerPolyhedron::dump() const { print(llvm::errs()); }
