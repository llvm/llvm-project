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

#ifndef MLIR_TESTPARAMETRICATTRIBUTES_H
#define MLIR_TESTPARAMETRICATTRIBUTES_H

#include <tuple>

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "TestParametricAttrInterfaces.h.inc"
#include "mlir/IR/DialectResourceBlobManager.h"

namespace testparametric {} // namespace testparametric

#define GET_ATTRDEF_CLASSES
#include "TestParametricAttrDefs.h.inc"

#endif // MLIR_TESTPARAMETRICATTRIBUTES_H
