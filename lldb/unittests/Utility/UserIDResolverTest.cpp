//===-- UserIDResolverTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/UserIDResolver.h"
#include "gmock/gmock.h"
#include <optional>

using namespace lldb_private;
using namespace testing;

namespace {
class TestUserIDResolver : public UserIDResolver {
public:
  MOCK_METHOD1(DoGetUserName, std::optional<std::string>(id_t uid));
  MOCK_METHOD1(DoGetGroupName, std::optional<std::string>(id_t gid));
};
} // namespace

TEST(UserIDResolver, GetUserName) {
  StrictMock<TestUserIDResolver> r;
  llvm::StringRef user47("foo");
  EXPECT_CALL(r, DoGetUserName(47)).Times(1).WillOnce(Return(user47.str()));
  EXPECT_CALL(r, DoGetUserName(42)).Times(1).WillOnce(Return(std::nullopt));

  // Call functions twice to make sure the caching works.
  EXPECT_EQ(user47, r.GetUserName(47));
  EXPECT_EQ(user47, r.GetUserName(47));
  EXPECT_EQ(std::nullopt, r.GetUserName(42));
  EXPECT_EQ(std::nullopt, r.GetUserName(42));
}

TEST(UserIDResolver, GetGroupName) {
  StrictMock<TestUserIDResolver> r;
  llvm::StringRef group47("foo");
  EXPECT_CALL(r, DoGetGroupName(47)).Times(1).WillOnce(Return(group47.str()));
  EXPECT_CALL(r, DoGetGroupName(42)).Times(1).WillOnce(Return(std::nullopt));

  // Call functions twice to make sure the caching works.
  EXPECT_EQ(group47, r.GetGroupName(47));
  EXPECT_EQ(group47, r.GetGroupName(47));
  EXPECT_EQ(std::nullopt, r.GetGroupName(42));
  EXPECT_EQ(std::nullopt, r.GetGroupName(42));
}
