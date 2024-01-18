//===- TestInterfaces.h - MLIR interfaces for testing -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares interfaces for the 'test' dialect that can be used for
// testing the interface infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H
#define MLIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {

class SpecializationParams {
public:
  SpecializationParams() {}

private:
  DenseMap<StringAttr, Attribute> params;
};

} // namespace mlir

#endif // MLIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H
