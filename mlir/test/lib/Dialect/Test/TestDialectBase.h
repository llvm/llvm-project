//===- TestDialectBase.h - The test dialect class ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares only the test dialect class, without its ops and the headers they
// need, for the compilation units of the attributes and types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTDIALECTBASE_H
#define MLIR_TEST_DIALECT_TEST_TESTDIALECTBASE_H

// The dialect declares a helper for the discardable attribute `SimpleAAttr`.
#include "TestAttributesCore.h"

namespace mlir {
class RewritePatternSet;
} // namespace mlir

#include "TestOpsDialect.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTDIALECTBASE_H
