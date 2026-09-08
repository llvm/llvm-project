//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/MCPError.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <system_error>

using namespace llvm;
using namespace lldb_protocol::mcp;

static std::string Log(const ErrorInfoBase &info) {
  std::string message;
  raw_string_ostream os(message);
  info.log(os);
  return message;
}

TEST(MCPErrorTest, DefaultErrorCode) {
  MCPError error("something went wrong");
  EXPECT_EQ(error.getMessage(), "something went wrong");
  EXPECT_EQ(Log(error), "something went wrong");

  std::error_code ec = error.convertToErrorCode();
  EXPECT_EQ(ec.value(), MCPError::kInternalError);
  EXPECT_EQ(ec.category(), std::generic_category());
}

TEST(MCPErrorTest, CustomErrorCode) {
  MCPError error("not found", MCPError::kResourceNotFound);
  EXPECT_EQ(error.getMessage(), "not found");

  std::error_code ec = error.convertToErrorCode();
  EXPECT_EQ(ec.value(), MCPError::kResourceNotFound);
  EXPECT_EQ(ec.category(), std::generic_category());
}

TEST(MCPErrorTest, AsLLVMError) {
  Error error = make_error<MCPError>("boom", 42);
  EXPECT_TRUE(error.isA<MCPError>());

  std::error_code ec = errorToErrorCode(std::move(error));
  EXPECT_EQ(ec.value(), 42);
}

TEST(MCPErrorTest, FailedWithMessage) {
  EXPECT_THAT_ERROR(make_error<MCPError>("explosion"),
                    FailedWithMessage("explosion"));
}

TEST(MCPErrorTest, UnsupportedURILog) {
  UnsupportedURI error("lldb://debugger/0");
  EXPECT_EQ(Log(error), "unsupported uri: lldb://debugger/0");
}

TEST(MCPErrorTest, UnsupportedURIErrorCode) {
  UnsupportedURI error("lldb://debugger/0");
  EXPECT_EQ(error.convertToErrorCode(), inconvertibleErrorCode());
}

TEST(MCPErrorTest, UnsupportedURIAsLLVMError) {
  Error error = make_error<UnsupportedURI>("lldb://foo");
  EXPECT_TRUE(error.isA<UnsupportedURI>());
  EXPECT_FALSE(error.isA<MCPError>());
  consumeError(std::move(error));
}
