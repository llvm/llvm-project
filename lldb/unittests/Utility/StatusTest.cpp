//===-- StatusTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace lldb_private;
using namespace lldb;

TEST(StatusTest, Formatv) {
  EXPECT_EQ("", llvm::formatv("{0}", Status()).str());
  EXPECT_EQ(
      "Hello Status",
      llvm::formatv("{0}", Status::FromErrorString("Hello Status")).str());
  EXPECT_EQ(
      "Hello",
      llvm::formatv("{0:5}", Status::FromErrorString("Hello Error")).str());
}

TEST(StatusTest, ErrorConstructor) {
  EXPECT_TRUE(Status::FromError(llvm::Error::success()).Success());

  Status eagain = Status::FromError(
      llvm::errorCodeToError(std::error_code(EAGAIN, std::generic_category())));
  EXPECT_TRUE(eagain.Fail());
  EXPECT_EQ(eErrorTypePOSIX, eagain.GetType());
  EXPECT_EQ(Status::ValueType(EAGAIN), eagain.GetError());

  Status foo = Status::FromError(llvm::createStringError("foo"));
  EXPECT_TRUE(foo.Fail());
  EXPECT_EQ(eErrorTypeGeneric, foo.GetType());
  EXPECT_STREQ("foo", foo.AsCString());

  foo = Status::FromError(llvm::Error::success());
  EXPECT_TRUE(foo.Success());
}

TEST(StatusTest, ErrorCodeConstructor) {
  EXPECT_TRUE(Status(std::error_code()).Success());

  Status eagain = std::error_code(EAGAIN, std::generic_category());
  EXPECT_TRUE(eagain.Fail());
  EXPECT_EQ(eErrorTypePOSIX, eagain.GetType());
  EXPECT_EQ(Status::ValueType(EAGAIN), eagain.GetError());

  llvm::Error list = llvm::joinErrors(llvm::createStringError("foo"),
                                      llvm::createStringError("bar"));
  Status foobar = Status::FromError(std::move(list));
  EXPECT_EQ(std::string("foo\nbar"), std::string(foobar.AsCString()));
}

TEST(StatusTest, ErrorConversion) {
  EXPECT_FALSE(bool(Status().ToError()));

  llvm::Error eagain = Status(EAGAIN, ErrorType::eErrorTypePOSIX).ToError();
  EXPECT_TRUE(bool(eagain));
  std::error_code ec = llvm::errorToErrorCode(std::move(eagain));
  EXPECT_EQ(EAGAIN, ec.value());
  EXPECT_EQ(std::generic_category(), ec.category());

  llvm::Error foo = Status::FromErrorString("foo").ToError();
  EXPECT_TRUE(bool(foo));
  EXPECT_EQ("foo", llvm::toString(std::move(foo)));
}

#ifdef _WIN32
TEST(StatusTest, ErrorWin32) {
  auto success = Status(NO_ERROR, ErrorType::eErrorTypeWin32);
  EXPECT_STREQ(NULL, success.AsCString());
  EXPECT_FALSE(success.ToError());
  EXPECT_TRUE(success.Success());

  WCHAR name[128]{};
  ULONG nameLen = std::size(name);
  ULONG langs = 0;
  GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &langs,
                              reinterpret_cast<PZZWSTR>(&name), &nameLen);
  // Skip the following tests on non-English, non-US, locales because the
  // formatted messages will be different.
  bool skip = wcscmp(L"en-US", name) != 0;

  Status s = Status(ERROR_ACCESS_DENIED, ErrorType::eErrorTypeWin32);
  EXPECT_TRUE(s.Fail());
  if (!skip)
    EXPECT_STREQ("Access is denied. ", s.AsCString());

  s = Status(ERROR_IPSEC_IKE_TIMED_OUT, ErrorType::eErrorTypeWin32);
  if (!skip)
    EXPECT_STREQ("Negotiation timed out ", s.AsCString());

  s = Status(16000, ErrorType::eErrorTypeWin32);
  if (!skip)
    EXPECT_STREQ("unknown error", s.AsCString());
}
#endif
