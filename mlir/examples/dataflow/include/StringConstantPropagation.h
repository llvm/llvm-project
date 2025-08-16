//===-- StringConstantPropagation.h - dataflow tutorial ---------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the dataflow tutorial's classes related to string
// constant propagation.
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_STRING_CONSTANT_PROPAGATION_H_
#define MLIR_TUTORIAL_STRING_CONSTANT_PROPAGATION_H_

#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace dataflow {

class StringConstant : public AnalysisState {
  /// This is the known string constant value of an SSA value at compile time
  /// as determined by a dataflow analysis. To implement the concept of being
  /// "uninitialized", the potential string value is wrapped in an `Optional`
  /// and set to `None` by default to indicate that no value has been provided.
  std::optional<std::string> stringValue = std::nullopt;

public:
  using AnalysisState::AnalysisState;

  /// Return true if no value has been provided for the string constant value.
  bool isUninitialized() const { return !stringValue.has_value(); }

  /// Default initialized the state to an empty string. Return whether the value
  /// of the state has changed.
  ChangeResult defaultInitialize() {
    // If the state already has a value, do nothing.
    if (!isUninitialized())
      return ChangeResult::NoChange;
    // Initialize the state and indicate that its value changed.
    stringValue = "";
    return ChangeResult::Change;
  }

  /// Get the currently known string value.
  StringRef getStringValue() const {
    assert(!isUninitialized() && "getting the value of an uninitialized state");
    return stringValue.value();
  }

  /// "Join" the value of the state with another constant.
  ChangeResult join(const Twine &value) {
    // If the current state is uninitialized, just take the value.
    if (isUninitialized()) {
      stringValue = value.str();
      return ChangeResult::Change;
    }
    // If the current state is "overdefined", no new information can be taken.
    if (stringValue->empty())
      return ChangeResult::NoChange;
    // If the current state has a different value, it now has two conflicting
    // values and should go to overdefined.
    if (stringValue != value.str()) {
      stringValue = "";
      return ChangeResult::Change;
    }
    return ChangeResult::NoChange;
  }

  /// Print the constant value.
  void print(raw_ostream &os) const override {
    os << stringValue.value_or("") << "\n";
  }
};

class StringConstantPropagation : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  /// Implement the transfer function for string operations. When visiting a
  /// string operation, this analysis will try to determine compile time values
  /// of the operation's results and set them in `StringConstant` states. This
  /// function is invoked on an operation whenever the states of its operands
  /// are changed.
  LogicalResult visit(ProgramPoint *point) override;

  /// Initialize the analysis by visiting every operation with potential
  /// control-flow semantics.
  LogicalResult initialize(Operation *top) override;
};

} // namespace dataflow
} // namespace mlir

#endif // MLIR_TUTORIAL_STRING_CONSTANT_PROPAGATION_H_
