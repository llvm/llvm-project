//===- DialectTest.cpp - Dialect unit tests -------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
struct TestDialect : public Dialect {
  TestDialect(MLIRContext *context) : Dialect(/*name=*/"test", context) {}
};

TEST(DialectDeathTest, MultipleDialectsWithSameNamespace) {
  MLIRContext context;

  // Registering a dialect with the same namespace twice should result in a
  // failure.
  new TestDialect(&context);
  ASSERT_DEATH(new TestDialect(&context), "");
}

} // end namespace
