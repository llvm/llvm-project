//===- TypeTest.cpp - SMT type unit tests ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace smt;

namespace {

TEST(SMTFuncTypeTest, NonEmptyDomain) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto boolTy = BoolType::get(&context);
  auto funcTy = SMTFuncType::getChecked(loc, ArrayRef<Type>{}, boolTy);
  ASSERT_EQ(funcTy, Type());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_STREQ(diag.str().c_str(), "domain must not be empty");
  });
}

} // namespace
