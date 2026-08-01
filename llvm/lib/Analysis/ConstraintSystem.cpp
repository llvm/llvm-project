//===- ConstraintSytem.cpp - A system of linear constraints. ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstraintSystem.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <string>

using namespace llvm;

#define DEBUG_TYPE "constraint-system"

bool ConstraintSystem::eliminateUsingFM() {
  // Implementation of Fourier–Motzkin elimination, with some tricks from the
  // paper Pugh, William. "The Omega test: a fast and practical integer
  // programming algorithm for dependence
  //  analysis."
  // Supercomputing'91: Proceedings of the 1991 ACM/
  // IEEE conference on Supercomputing. IEEE, 1991.
  assert(!Constraints.empty() &&
         "should only be called for non-empty constraint systems");

  unsigned LastIdx = NumVariables - 1;

  // First, either remove the variable in place if it is 0 or add the row to
  // RemainingRows and remove it from the system.
  SmallVector<SmallVector<Entry, 8>, 4> RemainingRows;
  for (unsigned R1 = 0; R1 < Constraints.size();) {
    SmallVector<Entry, 8> &Row1 = Constraints[R1];
    if (getLastCoefficient(Row1, LastIdx) == 0) {
      if (Row1.size() > 0 && Row1.back().Id == LastIdx)
        Row1.pop_back();
      R1++;
    } else {
      std::swap(Constraints[R1], Constraints.back());
      RemainingRows.push_back(std::move(Constraints.back()));
      Constraints.pop_back();
    }
  }

  // Process rows where the variable is != 0.
  unsigned NumRemainingConstraints = RemainingRows.size();
  for (unsigned R1 = 0; R1 < NumRemainingConstraints; R1++) {
    // FIXME do not use copy
    for (unsigned R2 = R1 + 1; R2 < NumRemainingConstraints; R2++) {
      // Examples of constraints stored as {Constant, Coeff_x, Coeff_y}
      // R1:  0 >=  1 * x + (-2) * y  => { 0,  1, -2 }
      // R2:  3 >=  2 * x +  3 * y    => { 3,  2,  3 }
      // LastIdx = 2 (tracking coefficient of y)
      // UpperLast: 3
      // LowerLast: -2
      int64_t UpperLast = getLastCoefficient(RemainingRows[R2], LastIdx);
      int64_t LowerLast = getLastCoefficient(RemainingRows[R1], LastIdx);
      assert(
          UpperLast != 0 && LowerLast != 0 &&
          "RemainingRows should only contain rows where the variable is != 0");

      if ((LowerLast < 0 && UpperLast < 0) || (LowerLast > 0 && UpperLast > 0))
        continue;

      unsigned LowerR = R1;
      unsigned UpperR = R2;
      if (UpperLast < 0) {
        std::swap(LowerR, UpperR);
        std::swap(LowerLast, UpperLast);
      }

      SmallVector<Entry, 8> NR;
      unsigned IdxUpper = 0;
      unsigned IdxLower = 0;
      auto &LowerRow = RemainingRows[LowerR];
      auto &UpperRow = RemainingRows[UpperR];
      // Combine the two rows to eliminate the variable. If any coefficient
      // computation overflows, skip them.
      bool Overflow = false;
      // Update constant and coefficients of both constraints.
      // Stops until every coefficient is updated or overflows.
      while (true) {
        if (IdxUpper >= UpperRow.size() || IdxLower >= LowerRow.size())
          break;
        int64_t M1, M2, N;
        // Starts with index 0 and updates every coefficients.
        int64_t UpperV = 0;
        int64_t LowerV = 0;
        uint16_t CurrentId = std::numeric_limits<uint16_t>::max();
        if (IdxUpper < UpperRow.size()) {
          CurrentId = std::min(UpperRow[IdxUpper].Id, CurrentId);
        }
        if (IdxLower < LowerRow.size()) {
          CurrentId = std::min(LowerRow[IdxLower].Id, CurrentId);
        }

        if (IdxUpper < UpperRow.size() && UpperRow[IdxUpper].Id == CurrentId) {
          UpperV = UpperRow[IdxUpper].Coefficient;
          IdxUpper++;
        }

        if (MulOverflow(UpperV, -1 * LowerLast, M1)) {
          Overflow = true;
          break;
        }
        if (IdxLower < LowerRow.size() && LowerRow[IdxLower].Id == CurrentId) {
          LowerV = LowerRow[IdxLower].Coefficient;
          IdxLower++;
        }

        if (MulOverflow(LowerV, UpperLast, M2)) {
          Overflow = true;
          break;
        }
        // This algorithm is a variant of sparse Gaussian elimination.
        //
        // The new coefficient for CurrentId is
        // N = UpperV * (-1) * LowerLast + LowerV * UpperLast
        //
        // UpperRow: { 3,  2,  3 }, LowerLast: -2
        // LowerRow: { 0,  1, -2 }, UpperLast: 3
        //
        // After multiplication:
        // UpperRow: { 6, 4, 6 }
        // LowerRow: { 0, 3, -6 }
        //
        // Eliminates y after addition:
        // N: { 6, 7, 0 } => 6 >= 7 * x
        if (AddOverflow(M1, M2, N)) {
          Overflow = true;
          break;
        }
        // Skip variable that is completely eliminated.
        if (N == 0)
          continue;
        NR.emplace_back(N, CurrentId);
      }
      if (Overflow || NR.empty())
        continue;
      Constraints.push_back(std::move(NR));
      // Give up if the new system gets too big.
      if (Constraints.size() > 500)
        return false;
    }
  }
  NumVariables -= 1;

  return true;
}

bool ConstraintSystem::mayHaveSolutionImpl() {
  while (!Constraints.empty() && NumVariables > 1) {
    if (!eliminateUsingFM())
      return true;
  }

  if (Constraints.empty() || NumVariables > 1)
    return true;

  return all_of(Constraints, [](auto &R) {
    if (R.empty())
      return true;
    if (R[0].Id == 0)
      return R[0].Coefficient >= 0;
    return true;
  });
}

SmallVector<std::string> ConstraintSystem::getVarNamesList() const {
  SmallVector<std::string> Names(Value2Index.size(), "");
#ifndef NDEBUG
  for (auto &[V, Index] : Value2Index) {
    std::string OperandName;
    if (V->getName().empty())
      OperandName = V->getNameOrAsOperand();
    else
      OperandName = std::string("%") + V->getName().str();
    Names[Index - 1] = OperandName;
  }
#endif
  return Names;
}

void ConstraintSystem::dump() const {
#ifndef NDEBUG
  if (Constraints.empty())
    return;
  SmallVector<std::string> Names = getVarNamesList();
  for (const auto &Row : Constraints) {
    SmallVector<std::string, 16> Parts;
    for (const Entry &E : Row) {
      if (E.Id >= NumVariables)
        break;
      if (E.Id == 0)
        continue;
      // The Value2Index map (and hence Names) may be absent, e.g. for the
      // temporary system solved in isConditionImplied. Fall back to a generic
      // variable name in that case.
      std::string Name = E.Id <= Names.size() ? Names[E.Id - 1]
                                              : ("%v" + std::to_string(E.Id));
      std::string Coefficient;
      if (E.Coefficient != 1)
        Coefficient = std::to_string(E.Coefficient) + " * ";
      Parts.push_back(Coefficient + Name);
    }
    // assert(!Parts.empty() && "need to have at least some parts");
    int64_t ConstPart = 0;
    if (Row[0].Id == 0)
      ConstPart = Row[0].Coefficient;
    LLVM_DEBUG(dbgs() << join(Parts, std::string(" + "))
                      << " <= " << std::to_string(ConstPart) << "\n");
  }
#endif
}

bool ConstraintSystem::mayHaveSolution() {
  LLVM_DEBUG(dbgs() << "---\n");
  LLVM_DEBUG(dump());
  bool HasSolution = mayHaveSolutionImpl();
  LLVM_DEBUG(dbgs() << (HasSolution ? "sat" : "unsat") << "\n");
  return HasSolution;
}

std::pair<ConstraintSystem, SmallVector<int64_t, 8>>
ConstraintSystem::getSubSystem(ArrayRef<int64_t> R) const {
  // Only constraints that share a variable (transitively) with a query R can
  // affect whether system + !R has a solution.
  //
  // Mark variables in the query and collect to the transitive closure over
  // variables that co-occur in a constraint row.
  ConstraintSystem SubSystem;
  SmallBitVector InSystem(NumVariables + 1, false);
  for (unsigned Id = 1, E = R.size(); Id < E; ++Id)
    if (R[Id] != 0)
      InSystem[Id] = true;
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &Row : Constraints) {
      // No common variables, skip.
      if (none_of(Row,
                  [&](const Entry &E) { return E.Id != 0 && InSystem[E.Id]; }))
        continue;
      for (const Entry &E : Row)
        if (E.Id != 0 && !InSystem[E.Id]) {
          InSystem[E.Id] = true;
          Changed = true;
        }
    }
  }

  // Assign compact indices to the variables of the sub-system.
  SmallVector<unsigned, 16> OldToNew;
  OldToNew.assign(NumVariables + 1, 0);
  unsigned NextIdx = 1;
  for (unsigned Id : InSystem.set_bits())
    OldToNew[Id] = NextIdx++;

  // Build new compact set of rows.
  SubSystem.NumVariables = NextIdx;
  for (const auto &Row : Constraints) {
    if (none_of(Row,
                [&](const Entry &E) { return E.Id != 0 && InSystem[E.Id]; }))
      continue;
    SmallVector<Entry, 8> NewRow;
    for (const Entry &E : Row) {
      if (!E.Id)
        NewRow.emplace_back(E.Coefficient, E.Id);
      else if (unsigned New = OldToNew[E.Id])
        NewRow.emplace_back(E.Coefficient, New);
    }
    SubSystem.Constraints.push_back(std::move(NewRow));
  }

  // Remap the query row into the component's compact index space.
  SmallVector<int64_t, 8> NewR(SubSystem.NumVariables, 0);
  NewR[0] = R[0];
  for (unsigned Id = 1, E = R.size(); Id < E; ++Id)
    if (R[Id] != 0)
      NewR[OldToNew[Id]] = R[Id];
  return {std::move(SubSystem), std::move(NewR)};
}

bool ConstraintSystem::isConditionImplied(SmallVector<int64_t, 8> R) const {
  // If all variable coefficients are 0, we have 'C >= 0'. If the constant is >=
  // 0, R is always true, regardless of the system.
  if (all_of(ArrayRef(R).drop_front(1), equal_to(0)))
    return R[0] >= 0;

  // If there is no solution with the negation of R added to the system, the
  // condition must hold based on the existing constraints.
  R = ConstraintSystem::negate(R);
  if (R.empty())
    return false;

  auto Copy = *this;
  Copy.addVariableRow(R);
  return !Copy.mayHaveSolution();
}

bool ConstraintSystem::isConditionImpliedInSubSystem(
    SmallVector<int64_t, 8> R) const {
  if (R.empty())
    return false;

  // Queries with no variables are trivially decided without building any
  // component.
  if (all_of(ArrayRef(R).drop_front(1), equal_to(0)))
    return R[0] >= 0;

  // A single query: build the component and solve it in place.
  const auto &[SubCS, NewR] = getSubSystem(R);
  return SubCS.isConditionImplied(NewR);
}
