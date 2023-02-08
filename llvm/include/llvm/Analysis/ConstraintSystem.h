//===- ConstraintSystem.h -  A system of linear constraints. --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
#define LLVM_ANALYSIS_CONSTRAINTSYSTEM_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace llvm {
class Value;
class ConstraintSystem {
  /// Current linear constraints in the system.
  /// An entry of the form c0, c1, ... cn represents the following constraint:
  ///   c0 >= v0 * c1 + .... + v{n-1} * cn
  SmallVector<SmallVector<int64_t, 8>, 4> Constraints;

  /// a Map between variabled collected in ConstraintElimination and their
  /// corresponding index in the constraint system solver.
  DenseMap<Value *, unsigned> Value2Index;

  /// Current greatest common divisor for all coefficients in the system.
  uint32_t GCD = 1;

  // Eliminate constraints from the system using Fourier–Motzkin elimination.
  bool eliminateUsingFM();

  /// Print the constraints in the system, using x0...xn as variable names.
  void dump() const;

  /// Returns true if there may be a solution for the constraints in the system.
  bool mayHaveSolutionImpl();

public:
  bool addVariableRow(ArrayRef<int64_t> R) {
    assert(Constraints.empty() || R.size() == Constraints.back().size());
    // If all variable coefficients are 0, the constraint does not provide any
    // usable information.
    if (all_of(ArrayRef(R).drop_front(1), [](int64_t C) { return C == 0; }))
      return false;

    for (const auto &C : R) {
      auto A = std::abs(C);
      GCD = APIntOps::GreatestCommonDivisor({32, (uint32_t)A}, {32, GCD})
                .getZExtValue();
    }
    Constraints.emplace_back(R.begin(), R.end());
    return true;
  }

  DenseMap<Value *, unsigned> &getValue2Index() { return Value2Index; }
  const DenseMap<Value *, unsigned> &getValue2Index() const {
    return Value2Index;
  }

  bool addVariableRowFill(ArrayRef<int64_t> R) {
    // If all variable coefficients are 0, the constraint does not provide any
    // usable information.
    if (all_of(ArrayRef(R).drop_front(1), [](int64_t C) { return C == 0; }))
      return false;

    for (auto &CR : Constraints) {
      while (CR.size() != R.size())
        CR.push_back(0);
    }
    return addVariableRow(R);
  }

  /// Returns true if there may be a solution for the constraints in the system.
  bool mayHaveSolution();

  static SmallVector<int64_t, 8> negate(SmallVector<int64_t, 8> R) {
    // The negated constraint R is obtained by multiplying by -1 and adding 1 to
    // the constant.
    R[0] += 1;
    for (auto &C : R)
      C *= -1;
    return R;
  }

  bool isConditionImplied(SmallVector<int64_t, 8> R) const;

  ArrayRef<int64_t> getLastConstraint() { return Constraints[0]; }
  void popLastConstraint() { Constraints.pop_back(); }
  void popLastNVariables(unsigned N) {
    for (auto &C : Constraints) {
      for (unsigned i = 0; i < N; i++)
        C.pop_back();
    }
  }

  /// Returns the number of rows in the constraint system.
  unsigned size() const { return Constraints.size(); }

  /// Print the constraints in the system, using \p Names as variable names.
  void dump(ArrayRef<std::string> Names) const;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
