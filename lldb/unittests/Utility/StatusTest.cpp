//===-- StatusTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"
#include "gtest/gtest.h"

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
