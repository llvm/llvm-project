//===- TestOpsSyntax.h - Operations for testing syntax ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSSYNTAX_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSSYNTAX_H

#include "TestAttributes.h"
#include "TestTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace test {
class TestReturnOp;
} // namespace test

#define GET_OP_CLASSES
#include "TestOpsSyntax.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSSYNTAX_H
