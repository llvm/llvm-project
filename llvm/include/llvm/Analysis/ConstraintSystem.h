//===- ConstraintSystem.h -  A system of linear constraints. --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
#define LLVM_ANALYSIS_CONSTRAINTSYSTEM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"

#include <string>

namespace llvm {

class Value;
class ConstraintSystem {
public:
  struct Entry {
    int64_t Coefficient;
    uint16_t Id;

    Entry(int64_t Coefficient, uint16_t Id)
        : Coefficient(Coefficient), Id(Id) {}
  };

  /// A single constraint of the form 'c >= v1 * c1 + ... + vn * cn'.
  using RowTy = SmallVector<Entry, 8>;

private:
  static int64_t getLastCoefficient(ArrayRef<Entry> R, uint16_t Id) {
    if (R.empty() || R.back().Id != Id)
      return 0;
    return R.back().Coefficient;
  }

  /// Returns true if \p R has an entry for the constant part.
  static bool hasConstantEntry(ArrayRef<Entry> R) {
    return !R.empty() && R.front().Id == 0;
  }

  /// Returns true if \p R does not have an entry for any variable, i.e. it is
  /// of the form 'c >= 0'.
  static bool isConstantOnly(ArrayRef<Entry> R) {
    return R.empty() || (R.size() == 1 && R.front().Id == 0);
  }

  /// Returns the constant part of \p R, which is 0 if \p R does not have an
  /// entry for it.
  static int64_t getConstant(ArrayRef<Entry> R) {
    return hasConstantEntry(R) ? R.front().Coefficient : 0;
  }

  /// Number of variables in the system, not counting the constant part. The
  /// variables use the indices 1 to NumVariables.
  size_t NumVariables = 0;

  /// Current linear constraints in the system.
  /// Each entry represents a constraint like
  ///   c0 >= v0 * c1 + .... + v{n-1} * cn
  SmallVector<RowTy, 4> Constraints;

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
  ConstraintSystem() = default;
  ConstraintSystem(ArrayRef<Value *> FunctionArgs) {
    NumVariables += FunctionArgs.size();
    for (auto *Arg : FunctionArgs) {
      Value2Index.insert({Arg, Value2Index.size() + 1});
    }
  }
  ConstraintSystem(const DenseMap<Value *, unsigned> &Value2Index)
      : NumVariables(Value2Index.size()), Value2Index(Value2Index) {}

  bool addRow(ArrayRef<Entry> R, size_t NumVars) {
    // If all variable coefficients are 0, the constraint does not provide any
    // usable information.
    if (isConstantOnly(R))
      return false;

    assert(NumVars >= R.back().Id && "NumVars must cover all variables in R");
    NumVariables = std::max(NumVars, NumVariables);
    // Only keep non-zero coefficients; in particular drop the entry for the
    // constant part if it is 0.
    RowTy &NewRow = Constraints.emplace_back();
    for (const Entry &E : R)
      if (E.Coefficient != 0)
        NewRow.push_back(E);
    return true;
  }

  DenseMap<Value *, unsigned> &getValue2Index() { return Value2Index; }
  const DenseMap<Value *, unsigned> &getValue2Index() const {
    return Value2Index;
  }

  /// Returns true if there may be a solution for the constraints in the system.
  LLVM_ABI bool mayHaveSolution();

  static RowTy negate(RowTy R) {
    assert(hasConstantEntry(R) && "row must have a constant entry");
    // The negated constraint R is obtained by multiplying by -1 and adding 1 to
    // the constant.
    if (AddOverflow(R[0].Coefficient, int64_t(1), R[0].Coefficient))
      return {};

    return negateOrEqual(std::move(R));
  }

  /// Multiplies each coefficient in the given row by -1. Returns an empty row
  /// on overflow. Does not modify the original row.
  ///
  /// \param R The row of coefficients to be negated.
  static RowTy negateOrEqual(RowTy R) {
    // The negated constraint R is obtained by multiplying by -1.
    for (Entry &E : R)
      if (MulOverflow(E.Coefficient, int64_t(-1), E.Coefficient))
        return {};
    return R;
  }

  /// Converts the given row to form a strict less than inequality. Returns an
  /// empty row on overflow. Does not modify the original row.
  ///
  /// \param R The row of coefficients to be converted.
  static RowTy toStrictLessThan(RowTy R) {
    assert(hasConstantEntry(R) && "row must have a constant entry");
    // The strict less than is obtained by subtracting 1 from the constant.
    if (SubOverflow(R[0].Coefficient, int64_t(1), R[0].Coefficient))
      return {};
    return R;
  }

  /// Build and return a sub-system of constraints connected (transitively) to
  /// query \p R, with variables compacted to a dense index range. Also
  /// translate \p R's entries to the sub-system.
  LLVM_ABI std::pair<ConstraintSystem, RowTy>
  getSubSystem(ArrayRef<Entry> R) const;

  LLVM_ABI bool isConditionImplied(RowTy R) const;
  LLVM_ABI bool isConditionImpliedInSubSystem(ArrayRef<Entry> R) const;

  const RowTy &getLastConstraint() const {
    assert(!Constraints.empty() && "Constraint system is empty");
    return Constraints.back();
  }

  void popLastConstraint() { Constraints.pop_back(); }
  void popLastNVariables(unsigned N) {
    assert(NumVariables >= N);
    NumVariables -= N;
  }

  /// Returns the number of rows in the constraint system.
  unsigned size() const { return Constraints.size(); }

  /// Print the constraints in the system.
  LLVM_ABI void dump() const;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_CONSTRAINTSYSTEM_H
