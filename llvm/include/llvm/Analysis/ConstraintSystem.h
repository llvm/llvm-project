//===- ConstraintSystem.h -  A system of linear constraints. --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
#define LLVM_ANALYSIS_CONSTRAINTSYSTEM_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Matrix.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

class Value;
class ConstraintSystem {
  struct Entry {
    int64_t Coefficient = 0;
    uint16_t Id = 0;

    Entry() = default;
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
  MatrixStorage<Entry, 64> Constraints;

  /// Constraints is only ever manipulated via this View.
  JaggedArrayView<Entry, 16, 64> View;

  /// A map of variables (IR values) to their corresponding index in the
  /// constraint system.
  DenseMap<Value *, unsigned> Value2Index;

  // Eliminate constraints from the system using Fourier–Motzkin elimination.
  bool eliminateUsingFM();

  /// Returns true if there may be a solution for the constraints in the system.
  bool mayHaveSolutionImpl();

  /// Get list of variable names from the Value2Index map.
  SmallVector<std::string> getVarNamesList() const;

public:
  // The Matrix Constraints should always be initialized with an upper-bound
  // number of columns. The default constructor hard-codes an upper-bound of 6,
  // as it is only used in unit tests, and not in the actual
  // ConstraintElimination Analysis.
  ConstraintSystem() : Constraints(6), View(Constraints) {}

  // This constructor is used by ConstraintElimination, inside ConstraintInfo.
  // Unfortunately, due to calls to addFact, that adds local variables, it is
  // impossible to know how many local variables there are in advance.
  // ConstraintElimination has a fixed upper-bound on the number of columns,
  // configurable as a cl::opt, so use that number, and don't add the constraint
  // if it exceeds that number.
  ConstraintSystem(ArrayRef<Value *> FunctionArgs, size_t NRows, size_t NCols)
      : Constraints(NCols), View(Constraints) {
    Constraints.reserve(NRows);
    NumVariables += FunctionArgs.size();
    for (auto *Arg : FunctionArgs) {
      Value2Index.insert({Arg, Value2Index.size() + 1});
    }
  }

  // This constructor is only used by the dump function in
  // ConstraintElimination.
  ConstraintSystem(const DenseMap<Value *, unsigned> &Value2Index,
                   unsigned NVars)
      : NumVariables(Value2Index.size()),
        Constraints(std::max(Value2Index.size(), NVars)), View(Constraints),
        Value2Index(Value2Index) {}

  ConstraintSystem(const ConstraintSystem &Other)
      : NumVariables(Other.NumVariables), Constraints(Other.Constraints),
        View(Other.View, Constraints), Value2Index(Other.Value2Index) {}

  bool addVariableRow(ArrayRef<int64_t> R) {
    assert(View.empty() || R.size() == NumVariables);
    // If all variable coefficients are 0, the constraint does not provide any
    // usable information.
    if (all_of(ArrayRef(R).drop_front(1), [](int64_t C) { return C == 0; }))
      return false;

    SmallVector<Entry, 4> NewRow;
    for (const auto &[Idx, C] : enumerate(R)) {
      if (C == 0)
        continue;
      NewRow.emplace_back(C, Idx);
    }

    // There is no correctness issue if we don't add a constraint.
    if (NewRow.size() > Constraints.getNumCols())
      return false;

    if (View.empty())
      NumVariables = R.size();

    View.addRow(std::move(NewRow));
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
    return negateOrEqual(R);
  }

  /// Multiplies each coefficient in the given vector by -1. Does not modify the
  /// original vector.
  ///
  /// \param R The vector of coefficients to be negated.
  static SmallVector<int64_t, 8> negateOrEqual(SmallVector<int64_t, 8> R) {
    // The negated constraint R is obtained by multiplying by -1.
    for (auto &C : R)
      if (MulOverflow(C, int64_t(-1), C))
        return {};
    return R;
  }

  /// Converts the given vector to form a strict less than inequality. Does not
  /// modify the original vector.
  ///
  /// \param R The vector of coefficients to be converted.
  static SmallVector<int64_t, 8> toStrictLessThan(SmallVector<int64_t, 8> R) {
    // The strict less than is obtained by subtracting 1 from the constant.
    if (SubOverflow(R[0], int64_t(1), R[0])) {
      return {};
    }
    return R;
  }

  bool isConditionImplied(SmallVector<int64_t, 8> R) const;

  SmallVector<int64_t> getLastConstraint() const {
    assert(!View.empty() && "Constraint system is empty");
    SmallVector<int64_t> Result(NumVariables, 0);
    for (auto &Entry : View.lastRow())
      Result[Entry.Id] = Entry.Coefficient;
    return Result;
  }

  void popLastConstraint() { View.dropLastRow(); }
  void popLastNVariables(unsigned N) {
    assert(NumVariables > N);
    NumVariables -= N;
  }

  /// Returns the number of rows in the constraint system.
  unsigned size() const { return View.getRowSpan(); }

  /// Print the constraints in the system.
  void dump() const;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
