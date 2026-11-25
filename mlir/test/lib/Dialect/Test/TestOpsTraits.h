//===- TestOpsTraits.h - Test dialect ops in partition `Traits` -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSTRAITS_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSTRAITS_H

#include "TestDialect.h"

#define GET_OP_CLASSES
#include "TestOpsTraits.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSTRAITS_H
