//===-- llvm/unittest/Debuginfod/HTTPClientTests.cpp - unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Support/Errc.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(BufferedHTTPResponseHandler, Lifecycle) {
  BufferedHTTPResponseHandler Handler;
  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length: 36\r\n"),
                    Succeeded());

  EXPECT_THAT_ERROR(Handler.handleBodyChunk("body:"), Succeeded());
  EXPECT_THAT_ERROR(Handler.handleBodyChunk("this puts the total at 36 chars"),
                    Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Body->MemoryBuffer::getBuffer(),
            "body:this puts the total at 36 chars");

  // Additional content should be rejected by the handler.
  EXPECT_THAT_ERROR(
      Handler.handleBodyChunk("extra content past the content-length"),
      Failed<llvm::StringError>());

  // Test response code is set.
  EXPECT_THAT_ERROR(Handler.handleStatusCode(200u), Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Code, 200u);
  EXPECT_THAT_ERROR(Handler.handleStatusCode(400u), Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Code, 400u);
}

TEST(BufferedHTTPResponseHandler, NoContentLengthLifecycle) {
  BufferedHTTPResponseHandler Handler;
  EXPECT_EQ(Handler.ResponseBuffer.Code, 0u);
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);

  // A body chunk passed before the content-length header is an error.
  EXPECT_THAT_ERROR(Handler.handleBodyChunk("body"),
                    Failed<llvm::StringError>());
  EXPECT_THAT_ERROR(Handler.handleHeaderLine("a header line"), Succeeded());
  EXPECT_THAT_ERROR(Handler.handleBodyChunk("body"),
                    Failed<llvm::StringError>());
}

TEST(BufferedHTTPResponseHandler, ZeroContentLength) {
  BufferedHTTPResponseHandler Handler;
  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length: 0"), Succeeded());
  EXPECT_NE(Handler.ResponseBuffer.Body, nullptr);
  EXPECT_EQ(Handler.ResponseBuffer.Body->getBufferSize(), 0u);

  // All content should be rejected by the handler.
  EXPECT_THAT_ERROR(Handler.handleBodyChunk("non-empty body content"),
                    Failed<llvm::StringError>());
}

TEST(BufferedHTTPResponseHandler, MalformedContentLength) {
  // Check that several invalid content lengths are ignored.
  BufferedHTTPResponseHandler Handler;
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);
  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length: fff"),
                    Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);

  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length:    "),
                    Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);

  using namespace std::string_literals;
  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length: \0\0\0"s),
                    Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);

  EXPECT_THAT_ERROR(Handler.handleHeaderLine("Content-Length: -11"),
                    Succeeded());
  EXPECT_EQ(Handler.ResponseBuffer.Body, nullptr);

  // All content should be rejected by the handler because no valid
  // Content-Length header has been received.
  EXPECT_THAT_ERROR(Handler.handleBodyChunk("non-empty body content"),
                    Failed<llvm::StringError>());
}

#ifdef LLVM_ENABLE_CURL

TEST(HTTPClient, isAvailable) { EXPECT_TRUE(HTTPClient::isAvailable()); }

#endif
