//===- TestOpsCore.h - Ops of TestOpsCore.td --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSCORE_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSCORE_H

#include "TestDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace test {
// Implicit terminators of `Core` ops, defined in other TestOps<Category>.td
// files.
class RegionYieldOp;
class TerminatorOp;
} // namespace test

#define GET_OP_CLASSES
#include "TestOpsCore.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSCORE_H
