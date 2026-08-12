//===-- ClangUserExpressionTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangUserExpression.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

namespace lldb_private {

/// ClangUserExpression declares this fixture a friend so that the tests below
/// can reach its private helpers.
///
/// TEST_F derives from this fixture and friendship isn't inherited, so the
/// private access has to happen in a member of the fixture itself rather than
/// in a test body.
class ClangUserExpressionTest : public testing::Test {
protected:
  static lldb::addr_t CallGetCppObjectPointer(lldb::StackFrameSP frame,
                                              llvm::StringRef object_name,
                                              Status &err) {
    return ClangUserExpression::GetCppObjectPointer(std::move(frame),
                                                    object_name, err);
  }
};

// GetCppObjectPointer must check the ValueObjectSP it gets back from
// GetObjectPointerValueObject before dereferencing it to look for a captured
// "this": that function returns a null SP when it can't find the object. A null
// frame is the cheapest way to make it do so.
TEST_F(ClangUserExpressionTest, GetCppObjectPointerWithoutFrame) {
  Status err;
  EXPECT_EQ(CallGetCppObjectPointer(nullptr, "this", err),
            LLDB_INVALID_ADDRESS);
  EXPECT_FALSE(err.Success());
}

} // namespace lldb_private
