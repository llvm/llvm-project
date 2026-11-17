//===- TestOpsAttributes.h - Test dialect ops in `Attributes` ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSATTRIBUTES_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSATTRIBUTES_H

#include "TestDialect.h"

#define GET_OP_CLASSES
#include "TestOpsAttributes.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSATTRIBUTES_H
