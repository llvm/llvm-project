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

#ifndef MLIR_TESTATTRIBUTES_H
#define MLIR_TESTATTRIBUTES_H

#include <tuple>

#include "TestTraits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "TestAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.h.inc"

#endif // MLIR_TESTATTRIBUTES_H
