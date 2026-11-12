//===- TestAttributesCore.h - Test attributes in `Core` ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESCORE_H
#define MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESCORE_H

#include <tuple>

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"

// generated files require above includes to come first
#include "TestAttrInterfaces.h.inc"
#include "TestOpEnums.h.inc"

namespace test {
class TestDialect;
// Payload class for the CopyCountAttr.
class CopyCount {
public:
  CopyCount(std::string value) : value(value) {}
  CopyCount(const CopyCount &rhs);
  CopyCount &operator=(const CopyCount &rhs);
  CopyCount(CopyCount &&rhs) = default;
  CopyCount &operator=(CopyCount &&rhs) = default;
  static int counter;
  std::string value;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const test::CopyCount &value);

/// A handle used to reference external elements instances.
using TestDialectResourceBlobHandle =
    mlir::DialectResourceBlobHandle<TestDialect>;
} // namespace test

#define GET_ATTRDEF_CLASSES
#include "TestAttributesCore.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTATTRIBUTESCORE_H
