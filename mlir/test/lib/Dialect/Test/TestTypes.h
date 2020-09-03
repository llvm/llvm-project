//===- TestTypes.h - MLIR Test Dialect Types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTYPES_H
#define MLIR_TESTTYPES_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

namespace mlir {

#include "TestTypeInterfaces.h.inc"

/// This class is a simple test type that uses a generated interface.
struct TestType : public Type::TypeBase<TestType, Type, TypeStorage,
                                        TestTypeInterface::Trait> {
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_9_TYPE;
  }

  static TestType get(MLIRContext *context) {
    return Base::get(context, Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_9_TYPE);
  }

  /// Provide a definition for the necessary interface methods.
  void printTypeC(Location loc) const {
    emitRemark(loc) << *this << " - TestC";
  }
};
} // end namespace mlir

#endif // MLIR_TESTTYPES_H
