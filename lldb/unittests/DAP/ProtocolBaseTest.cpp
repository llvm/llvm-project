//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolBase.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include <gtest/gtest.h>

using namespace llvm;
using namespace lldb_dap::protocol;
using lldb_private::PrettyPrint;
using llvm::json::parse;
using llvm::json::Value;

TEST(ProtocolBaseTest, SanitizedString) {
  for (auto [input, json] : std::vector<std::pair<const char *, const char *>>{
           {"valid str", R"("valid str")"},
           {"lone trailing \x81\x82 bytes", R"("lone trailing �� bytes")"}}) {
    String str = input;
    Expected<Value> expected_str = parse(json);
    ASSERT_THAT_EXPECTED(expected_str, llvm::Succeeded());
    EXPECT_EQ(PrettyPrint(*expected_str), PrettyPrint(str));
  }
}
