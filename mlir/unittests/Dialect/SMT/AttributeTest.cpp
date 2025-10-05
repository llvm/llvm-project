//===- AttributeTest.cpp - SMT attribute unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SMT/IR/SMTAttributes.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace smt;

namespace {

TEST(BitVectorAttrTest, MinBitWidth) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr = BitVectorAttr::getChecked(loc, &context, UINT64_C(0), 0U);
  ASSERT_EQ(attr, BitVectorAttr());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "bit-width must be at least 1, but got 0");
  });
}

TEST(BitVectorAttrTest, ParserAndPrinterCorrect) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();

  auto attr = BitVectorAttr::get(&context, "#b1010");
  ASSERT_EQ(attr.getValue(), APInt(4, 10));
  ASSERT_EQ(attr.getType(), BitVectorType::get(&context, 4));

  // A bit-width divisible by 4 is always printed in hex
  attr = BitVectorAttr::get(&context, "#b01011010");
  ASSERT_EQ(attr.getValueAsString(), "#x5a");

  // A bit-width not divisible by 4 is always printed in binary
  // Also, make sure leading zeros are printed
  attr = BitVectorAttr::get(&context, "#b0101101");
  ASSERT_EQ(attr.getValueAsString(), "#b0101101");

  attr = BitVectorAttr::get(&context, "#x3c");
  ASSERT_EQ(attr.getValueAsString(), "#x3c");

  attr = BitVectorAttr::get(&context, "#x03c");
  ASSERT_EQ(attr.getValueAsString(), "#x03c");
}

TEST(BitVectorAttrTest, ExpectedOneDigit) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr =
      BitVectorAttr::getChecked(loc, &context, static_cast<StringRef>("#b"));
  ASSERT_EQ(attr, BitVectorAttr());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "expected at least one digit");
  });
}

TEST(BitVectorAttrTest, ExpectedBOrX) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr =
      BitVectorAttr::getChecked(loc, &context, static_cast<StringRef>("#c0"));
  ASSERT_EQ(attr, BitVectorAttr());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "expected either 'b' or 'x'");
  });
}

TEST(BitVectorAttrTest, ExpectedHashtag) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr =
      BitVectorAttr::getChecked(loc, &context, static_cast<StringRef>("b0"));
  ASSERT_EQ(attr, BitVectorAttr());
  context.getDiagEngine().registerHandler(
      [&](Diagnostic &diag) { ASSERT_EQ(diag.str(), "expected '#'"); });
}

TEST(BitVectorAttrTest, OutOfRange) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  auto attr1 = BitVectorAttr::getChecked(loc, &context, UINT64_C(2), 1U);
  auto attr63 =
      BitVectorAttr::getChecked(loc, &context, UINT64_C(3) << 62, 63U);
  ASSERT_EQ(attr1, BitVectorAttr());
  ASSERT_EQ(attr63, BitVectorAttr());
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(),
              "value does not fit in a bit-vector of desired width");
  });
}

TEST(BitVectorAttrTest, GetUInt64Max) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  auto attr64 = BitVectorAttr::get(&context, UINT64_MAX, 64);
  auto attr65 = BitVectorAttr::get(&context, UINT64_MAX, 65);
  ASSERT_EQ(attr64.getValue(), APInt::getAllOnes(64));
  ASSERT_EQ(attr65.getValue(), APInt::getAllOnes(64).zext(65));
}

} // namespace
