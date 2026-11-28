//===- TestAttributesSyntax.h - Test attributes in `Syntax` -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESSYNTAX_H
#define MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESSYNTAX_H

// The attributes in `Syntax` use the enums and attributes in `Core`.
#include "TestAttributesCore.h"

#define GET_ATTRDEF_CLASSES
#include "TestAttributesSyntax.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESSYNTAX_H
