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
  struct Entry {
    int64_t Coefficient;
    uint16_t Id;

    Entry(int64_t Coefficient, uint16_t Id)
        : Coefficient(Coefficient), Id(Id) {}
  };

  static int64_t getConstPart(const Entry &E) {
    if (E.Id == 0)
      return E.Coefficient;
    return 0;
  }

  static int64_t getLastCoefficient(ArrayRef<Entry> Row, uint16_t Id) {
    if (Row.empty())
      return 0;
    if (Row.back().Id == Id)
      return Row.back().Coefficient;
    return 0;
  }

  size_t NumVariables = 0;

  /// Current linear constraints in the system.
  /// An entry of the form c0, c1, ... cn represents the following constraint:
  ///   c0 >= v0 * c1 + .... + v{n-1} * cn
  SmallVector<SmallVector<Entry, 8>, 4> Constraints;

  /// A map of variables (IR values) to their corresponding index in the
  /// constraint system.
  DenseMap<Value *, unsigned> Value2Index;

  /// Current greatest common divisor for all coefficients in the system.
  uint32_t GCD = 1;

  // Eliminate constraints from the system using Fourier–Motzkin elimination.
  bool eliminateUsingFM();

  /// Returns true if there may be a solution for the constraints in the system.
  bool mayHaveSolutionImpl();

  /// Get list of variable names from the Value2Index map.
  SmallVector<std::string> getVarNamesList() const;

public:
  ConstraintSystem() {}
  ConstraintSystem(const DenseMap<Value *, unsigned> &Value2Index)
      : Value2Index(Value2Index) {}

  bool addVariableRow(ArrayRef<int64_t> R) {
    assert(Constraints.empty() || R.size() == NumVariables);
    // If all variable coefficients are 0, the constraint does not provide any
    // usable information.
    if (all_of(ArrayRef(R).drop_front(1), [](int64_t C) { return C == 0; }))
      return false;

    SmallVector<Entry, 4> NewRow;
    for (const auto &[Idx, C] : enumerate(R)) {
      if (C == 0)
        continue;
      auto A = std::abs(C);
      GCD = APIntOps::GreatestCommonDivisor({32, (uint32_t)A}, {32, GCD})
                .getZExtValue();

      NewRow.emplace_back(C, Idx);
    }
    if (Constraints.empty())
      NumVariables = R.size();
    Constraints.push_back(std::move(NewRow));
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

    NumVariables = std::max(R.size(), NumVariables);
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

  SmallVector<int64_t> getLastConstraint() const {
    assert(!Constraints.empty() && "Constraint system is empty");
    SmallVector<int64_t> Result(NumVariables, 0);
    for (auto &Entry : Constraints.back())
      Result[Entry.Id] = Entry.Coefficient;
    return Result;
  }

  void popLastConstraint() { Constraints.pop_back(); }
  void popLastNVariables(unsigned N) {
    assert(NumVariables > N);
    NumVariables -= N;
  }

  /// Returns the number of rows in the constraint system.
  unsigned size() const { return Constraints.size(); }

  /// Print the constraints in the system.
  void dump() const;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
