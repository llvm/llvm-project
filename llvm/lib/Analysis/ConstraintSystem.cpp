//===- ConstraintSytem.cpp - A system of linear constraints. ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstraintSystem.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"

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
  unsigned NumVariables = Constraints[0].size();

  uint32_t NewGCD = 1;
  unsigned LastIdx = NumVariables - 1;

  // First, either remove the variable in place if it is 0 or add the row to
  // RemainingRows and remove it from the system.
  SmallVector<SmallVector<int64_t, 8>, 4> RemainingRows;
  for (unsigned R1 = 0; R1 < Constraints.size();) {
    SmallVector<int64_t, 8> &Row1 = Constraints[R1];
    int64_t LowerLast = Row1[LastIdx];
    if (LowerLast == 0) {
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
      if (R1 == R2)
        continue;

      int64_t UpperLast = RemainingRows[R2][LastIdx];
      int64_t LowerLast = RemainingRows[R1][LastIdx];
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

      SmallVector<int64_t, 8> NR;
      for (unsigned I = 0; I < LastIdx; I++) {
        int64_t M1, M2, N;
        int64_t UpperV = RemainingRows[UpperR][I];
        if (MulOverflow(UpperV, ((-1) * LowerLast / GCD), M1))
          return false;
        int64_t LowerV = RemainingRows[LowerR][I];
        if (MulOverflow(LowerV, (UpperLast / GCD), M2))
          return false;
        if (AddOverflow(M1, M2, N))
          return false;
        NR.push_back(N);

        NewGCD =
            APIntOps::GreatestCommonDivisor({32, (uint32_t)N}, {32, NewGCD})
                .getZExtValue();
      }
      Constraints.push_back(std::move(NR));
      // Give up if the new system gets too big.
      if (Constraints.size() > 500)
        return false;
    }
  }
  GCD = NewGCD;

  return true;
}

bool ConstraintSystem::mayHaveSolutionImpl() {
  while (!Constraints.empty() && Constraints[0].size() > 1) {
    if (!eliminateUsingFM())
      return true;
  }

  if (Constraints.empty() || Constraints[0].size() > 1)
    return true;

  return all_of(Constraints, [](auto &R) { return R[0] >= 0; });
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
    for (unsigned I = 1, S = Row.size(); I < S; ++I) {
      if (Row[I] == 0)
        continue;
      std::string Coefficient;
      if (Row[I] != 1)
        Coefficient = std::to_string(Row[I]) + " * ";
      Parts.push_back(Coefficient + Names[I - 1]);
    }
    assert(!Parts.empty() && "need to have at least some parts");
    LLVM_DEBUG(dbgs() << join(Parts, std::string(" + "))
                      << " <= " << std::to_string(Row[0]) << "\n");
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

bool ConstraintSystem::isConditionImplied(SmallVector<int64_t, 8> R) const {
  // If all variable coefficients are 0, we have 'C >= 0'. If the constant is >=
  // 0, R is always true, regardless of the system.
  if (all_of(ArrayRef(R).drop_front(1), [](int64_t C) { return C == 0; }))
    return R[0] >= 0;

  // If there is no solution with the negation of R added to the system, the
  // condition must hold based on the existing constraints.
  R = ConstraintSystem::negate(R);

  auto NewSystem = *this;
  NewSystem.addVariableRow(R);
  return !NewSystem.mayHaveSolution();
}
