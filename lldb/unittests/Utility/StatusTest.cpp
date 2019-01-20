//===-- StatusTest.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#include <winerror.h>
#endif

using namespace lldb_private;
using namespace lldb;

TEST(StatusTest, Formatv) {
  EXPECT_EQ("", llvm::formatv("{0}", Status()).str());
  EXPECT_EQ("Hello Status", llvm::formatv("{0}", Status("Hello Status")).str());
  EXPECT_EQ("Hello", llvm::formatv("{0:5}", Status("Hello Error")).str());
}

TEST(StatusTest, ErrorConstructor) {
  EXPECT_TRUE(Status(llvm::Error::success()).Success());

  Status eagain(
      llvm::errorCodeToError(std::error_code(EAGAIN, std::generic_category())));
  EXPECT_TRUE(eagain.Fail());
  EXPECT_EQ(eErrorTypePOSIX, eagain.GetType());
  EXPECT_EQ(Status::ValueType(EAGAIN), eagain.GetError());

  Status foo(llvm::make_error<llvm::StringError>(
      "foo", llvm::inconvertibleErrorCode()));
  EXPECT_TRUE(foo.Fail());
  EXPECT_EQ(eErrorTypeGeneric, foo.GetType());
  EXPECT_STREQ("foo", foo.AsCString());

  foo = llvm::Error::success();
  EXPECT_TRUE(foo.Success());
}

TEST(StatusTest, ErrorConversion) {
  EXPECT_FALSE(bool(Status().ToError()));

  llvm::Error eagain = Status(EAGAIN, ErrorType::eErrorTypePOSIX).ToError();
  EXPECT_TRUE(bool(eagain));
  std::error_code ec = llvm::errorToErrorCode(std::move(eagain));
  EXPECT_EQ(EAGAIN, ec.value());
  EXPECT_EQ(std::generic_category(), ec.category());

  llvm::Error foo = Status("foo").ToError();
  EXPECT_TRUE(bool(foo));
  EXPECT_EQ("foo", llvm::toString(std::move(foo)));
}

#ifdef _WIN32
TEST(StatusTest, ErrorWin32) {
  auto success = Status(NO_ERROR, ErrorType::eErrorTypeWin32);
  EXPECT_STREQ(NULL, success.AsCString());
  EXPECT_FALSE(success.ToError());
  EXPECT_TRUE(success.Success());

  auto s = Status(ERROR_ACCESS_DENIED, ErrorType::eErrorTypeWin32);
  EXPECT_TRUE(s.Fail());
  EXPECT_STREQ("Access is denied. ", s.AsCString());

  s.SetError(ERROR_IPSEC_IKE_TIMED_OUT, ErrorType::eErrorTypeWin32);
  EXPECT_STREQ("Negotiation timed out ", s.AsCString());

  s.SetError(16000, ErrorType::eErrorTypeWin32);
  EXPECT_STREQ("unknown error", s.AsCString());
}
#endif
