//===- InstructionCost.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines an InstructionCost class that is used when calculating
/// the cost of an instruction, or a group of instructions. In addition to a
/// numeric value representing the cost the class also contains a state that
/// can be used to encode particular properties, i.e. a cost being invalid or
/// unknown.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_INSTRUCTIONCOST_H
#define LLVM_SUPPORT_INSTRUCTIONCOST_H

#include "llvm/ADT/Optional.h"

namespace llvm {

class raw_ostream;

class InstructionCost {
public:
  using CostType = int;

  /// These states can currently be used to indicate whether a cost is valid or
  /// invalid. Examples of an invalid cost might be where the cost is
  /// prohibitively expensive and the user wants to prevent certain
  /// optimizations being performed. Or perhaps the cost is simply unknown
  /// because the operation makes no sense in certain circumstances. These
  /// states can be expanded in future to support other cases if necessary.
  enum CostState { Valid, Invalid };

private:
  CostType Value;
  CostState State;

  void propagateState(const InstructionCost &RHS) {
    if (RHS.State == Invalid)
      State = Invalid;
  }

public:
  InstructionCost() = default;

  InstructionCost(CostState) = delete;
  InstructionCost(CostType Val) : Value(Val), State(Valid) {}

  static InstructionCost getInvalid(CostType Val = 0) {
    InstructionCost Tmp(Val);
    Tmp.setInvalid();
    return Tmp;
  }

  bool isValid() const { return State == Valid; }
  void setValid() { State = Valid; }
  void setInvalid() { State = Invalid; }
  CostState getState() const { return State; }

  /// This function is intended to be used as sparingly as possible, since the
  /// class provides the full range of operator support required for arithmetic
  /// and comparisons.
  Optional<CostType> getValue() const {
    if (isValid())
      return Value;
    return None;
  }

  /// For all of the arithmetic operators provided here any invalid state is
  /// perpetuated and cannot be removed. Once a cost becomes invalid it stays
  /// invalid, and it also inherits any invalid state from the RHS. Regardless
  /// of the state, arithmetic and comparisons work on the actual values in the
  /// same way as they would on a basic type, such as integer.

  InstructionCost &operator+=(const InstructionCost &RHS) {
    propagateState(RHS);
    Value += RHS.Value;
    return *this;
  }

  InstructionCost &operator+=(const CostType RHS) {
    InstructionCost RHS2(RHS);
    *this += RHS2;
    return *this;
  }

  InstructionCost &operator-=(const InstructionCost &RHS) {
    propagateState(RHS);
    Value -= RHS.Value;
    return *this;
  }

  InstructionCost &operator-=(const CostType RHS) {
    InstructionCost RHS2(RHS);
    *this -= RHS2;
    return *this;
  }

  InstructionCost &operator*=(const InstructionCost &RHS) {
    propagateState(RHS);
    Value *= RHS.Value;
    return *this;
  }

  InstructionCost &operator*=(const CostType RHS) {
    InstructionCost RHS2(RHS);
    *this *= RHS2;
    return *this;
  }

  InstructionCost &operator/=(const InstructionCost &RHS) {
    propagateState(RHS);
    Value /= RHS.Value;
    return *this;
  }

  InstructionCost &operator/=(const CostType RHS) {
    InstructionCost RHS2(RHS);
    *this /= RHS2;
    return *this;
  }

  InstructionCost &operator++() {
    *this += 1;
    return *this;
  }

  InstructionCost operator++(int) {
    InstructionCost Copy = *this;
    ++*this;
    return Copy;
  }

  InstructionCost &operator--() {
    *this -= 1;
    return *this;
  }

  InstructionCost operator--(int) {
    InstructionCost Copy = *this;
    --*this;
    return Copy;
  }

  bool operator==(const InstructionCost &RHS) const {
    return State == RHS.State && Value == RHS.Value;
  }

  bool operator!=(const InstructionCost &RHS) const { return !(*this == RHS); }

  bool operator==(const CostType RHS) const {
    return State == Valid && Value == RHS;
  }

  bool operator!=(const CostType RHS) const { return !(*this == RHS); }

  /// For the comparison operators we have chosen to use total ordering with
  /// the following rules:
  ///  1. If either of the states != Valid then a lexicographical order is
  ///     applied based upon the state.
  ///  2. If both states are valid then order based upon value.
  /// This avoids having to add asserts the comparison operators that the states
  /// are valid and users can test for validity of the cost explicitly.
  bool operator<(const InstructionCost &RHS) const {
    if (State != Valid || RHS.State != Valid)
      return State < RHS.State;
    return Value < RHS.Value;
  }

  bool operator>(const InstructionCost &RHS) const { return RHS < *this; }

  bool operator<=(const InstructionCost &RHS) const { return !(RHS < *this); }

  bool operator>=(const InstructionCost &RHS) const { return !(*this < RHS); }

  bool operator<(const CostType RHS) const {
    InstructionCost RHS2(RHS);
    return *this < RHS2;
  }

  bool operator>(const CostType RHS) const {
    InstructionCost RHS2(RHS);
    return *this > RHS2;
  }

  bool operator<=(const CostType RHS) const {
    InstructionCost RHS2(RHS);
    return *this <= RHS2;
  }

  bool operator>=(const CostType RHS) const {
    InstructionCost RHS2(RHS);
    return *this >= RHS2;
  }

  void print(raw_ostream &OS) const;
};

inline InstructionCost operator+(const InstructionCost &LHS,
                                 const InstructionCost &RHS) {
  InstructionCost LHS2(LHS);
  LHS2 += RHS;
  return LHS2;
}

inline InstructionCost operator-(const InstructionCost &LHS,
                                 const InstructionCost &RHS) {
  InstructionCost LHS2(LHS);
  LHS2 -= RHS;
  return LHS2;
}

inline InstructionCost operator*(const InstructionCost &LHS,
                                 const InstructionCost &RHS) {
  InstructionCost LHS2(LHS);
  LHS2 *= RHS;
  return LHS2;
}

inline InstructionCost operator/(const InstructionCost &LHS,
                                 const InstructionCost &RHS) {
  InstructionCost LHS2(LHS);
  LHS2 /= RHS;
  return LHS2;
}

inline raw_ostream &operator<<(raw_ostream &OS, const InstructionCost &V) {
  V.print(OS);
  return OS;
}

} // namespace llvm

#endif
