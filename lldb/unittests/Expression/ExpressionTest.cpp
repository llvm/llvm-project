//===-- ExpressionTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "TestingSupport/TestUtilities.h"
#include "lldb/Expression/Expression.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;

struct LabelTestCase {
  llvm::StringRef encoded;
  FunctionCallLabel label;
  llvm::SmallVector<llvm::StringRef> error_pattern;
};

static LabelTestCase g_label_test_cases[] = {
    // Failure modes
    {"bar:blah:0x0:0x0:_Z3foov",
     {},
     {"expected function call label prefix '$__lldb_func' but found 'bar' "
      "instead."}},
    {"$__lldb_func :blah:0x0:0x0:_Z3foov",
     {},
     {"expected function call label prefix '$__lldb_func' but found "
      "'$__lldb_func ' instead."}},
    {"$__lldb_funcc:blah:0x0:0x0:_Z3foov",
     {},
     {"expected function call label prefix '$__lldb_func' but found "
      "'$__lldb_funcc' instead."}},
    {"", {}, {"malformed function call label."}},
    {"foo", {}, {"malformed function call label."}},
    {"$__lldb_func", {}, {"malformed function call label."}},
    {"$__lldb_func:", {}, {"malformed function call label."}},
    {"$__lldb_func:blah", {}, {"malformed function call label."}},
    {"$__lldb_func:blah:0x0", {}, {"malformed function call label."}},
    {"$__lldb_func:111:0x0:0x0", {}, {"malformed function call label."}},
    {"$__lldb_func:111:abc:0x0:_Z3foov",
     {},
     {"failed to parse module ID from 'abc'."}},
    {"$__lldb_func:111:-1:0x0:_Z3foov",
     {},
     {"failed to parse module ID from '-1'."}},
    {"$__lldb_func:111:0x0invalid:0x0:_Z3foov",
     {},
     {"failed to parse module ID from '0x0invalid'."}},
    {"$__lldb_func:111:0x0 :0x0:_Z3foov",
     {},
     {"failed to parse module ID from '0x0 '."}},
    {"$__lldb_func:blah:0x0:abc:_Z3foov",
     {},
     {"failed to parse symbol ID from 'abc'."}},
    {"$__lldb_func:blah:0x5:-1:_Z3foov",
     {},
     {"failed to parse symbol ID from '-1'."}},
    {"$__lldb_func:blah:0x5:0x0invalid:_Z3foov",
     {},
     {"failed to parse symbol ID from '0x0invalid'."}},
    {"$__lldb_func:blah:0x5:0x0 :_Z3foov",
     {},
     {"failed to parse symbol ID from '0x0 '."}},
    {"$__lldb_func:blah:0x0:0x0:_Z3foov",
     {
         /*.discriminator=*/"blah",
         /*.module_id=*/0x0,
         /*.symbol_id=*/0x0,
         /*.lookup_name=*/"_Z3foov",
     },
     {}},
    {"$__lldb_func::0x0:0x0:abc:def:::a",
     {
         /*.discriminator=*/"",
         /*.module_id=*/0x0,
         /*.symbol_id=*/0x0,
         /*.lookup_name=*/"abc:def:::a",
     },
     {}},
    {"$__lldb_func:0x45:0xd2:0xf0:$__lldb_func",
     {
         /*.discriminator=*/"0x45",
         /*.module_id=*/0xd2,
         /*.symbol_id=*/0xf0,
         /*.lookup_name=*/"$__lldb_func",
     },
     {}},
};

struct ExpressionTestFixture : public testing::TestWithParam<LabelTestCase> {};

TEST_P(ExpressionTestFixture, FunctionCallLabel) {
  const auto &[encoded, label, errors] = GetParam();

  auto decoded_or_err = FunctionCallLabel::fromString(encoded);
  if (!errors.empty()) {
    EXPECT_THAT_EXPECTED(
        decoded_or_err,
        llvm::FailedWithMessageArray(testing::ElementsAreArray(errors)));
    return;
  }

  EXPECT_THAT_EXPECTED(decoded_or_err, llvm::Succeeded());

  auto label_str = label.toString();
  EXPECT_EQ(decoded_or_err->toString(), encoded);
  EXPECT_EQ(label_str, encoded);

  EXPECT_EQ(decoded_or_err->discriminator, label.discriminator);
  EXPECT_EQ(decoded_or_err->module_id, label.module_id);
  EXPECT_EQ(decoded_or_err->symbol_id, label.symbol_id);
  EXPECT_EQ(decoded_or_err->lookup_name, label.lookup_name);

  auto roundtrip_or_err = FunctionCallLabel::fromString(label_str);
  EXPECT_THAT_EXPECTED(roundtrip_or_err, llvm::Succeeded());

  EXPECT_EQ(roundtrip_or_err->discriminator, label.discriminator);
  EXPECT_EQ(roundtrip_or_err->module_id, label.module_id);
  EXPECT_EQ(roundtrip_or_err->symbol_id, label.symbol_id);
  EXPECT_EQ(roundtrip_or_err->lookup_name, label.lookup_name);
}

INSTANTIATE_TEST_SUITE_P(FunctionCallLabelTest, ExpressionTestFixture,
                         testing::ValuesIn(g_label_test_cases));
