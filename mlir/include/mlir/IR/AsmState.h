//===- AsmState.h - State class for AsmPrinter ------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AsmState class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ASMSTATE_H_
#define MLIR_IR_ASMSTATE_H_

#include <memory>

namespace mlir {
class Operation;

namespace detail {
class AsmStateImpl;
} // end namespace detail

/// This class provides management for the lifetime of the state used when
/// printing the IR. It allows for alleviating the cost of recomputing the
/// internal state of the asm printer.
///
/// The IR should not be mutated in-between invocations using this state, and
/// the IR being printed must not be an parent of the IR originally used to
/// initialize this state. This means that if a child operation is provided, a
/// parent operation cannot reuse this state.
class AsmState {
public:
  /// Initialize the asm state at the level of the given operation.
  AsmState(Operation *op);
  ~AsmState();

  /// Return an instance of the internal implementation. Returns nullptr if the
  /// state has not been initialized.
  detail::AsmStateImpl &getImpl() { return *impl; }

private:
  AsmState() = delete;

  /// A pointer to allocated storage for the impl state.
  std::unique_ptr<detail::AsmStateImpl> impl;
};

} // end namespace mlir

#endif // MLIR_IR_ASMSTATE_H_
