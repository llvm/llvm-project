//===- TestAttributes.cpp - MLIR Test Dialect Attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains attributes defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestParametricAttributes.h"
#include "TestParametricDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace testparametric;

//===----------------------------------------------------------------------===//
// TestParametricDialect
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "TestParametricAttrDefs.cpp.inc"

void TestParametricDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestParametricAttrDefs.cpp.inc"
      >();
}
