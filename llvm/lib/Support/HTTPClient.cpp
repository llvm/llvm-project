//===-- llvm/Support/HTTPClient.cpp - HTTP client library -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file defines the methods of the HTTPRequest, HTTPClient, and
/// BufferedHTTPResponseHandler classes.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/HTTPClient.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

HTTPRequest::HTTPRequest(StringRef Url) { this->Url = Url.str(); }

bool operator==(const HTTPRequest &A, const HTTPRequest &B) {
  return A.Url == B.Url && A.Method == B.Method &&
         A.FollowRedirects == B.FollowRedirects;
}

HTTPResponseHandler::~HTTPResponseHandler() = default;

static inline bool parseContentLengthHeader(StringRef LineRef,
                                            size_t &ContentLength) {
  // Content-Length is a mandatory header, and the only one we handle.
  return LineRef.consume_front("Content-Length: ") &&
         to_integer(LineRef.trim(), ContentLength, 10);
}

Error BufferedHTTPResponseHandler::handleHeaderLine(StringRef HeaderLine) {
  if (ResponseBuffer.Body)
    return Error::success();

  size_t ContentLength;
  if (parseContentLengthHeader(HeaderLine, ContentLength))
    ResponseBuffer.Body =
        WritableMemoryBuffer::getNewUninitMemBuffer(ContentLength);

  return Error::success();
}

Error BufferedHTTPResponseHandler::handleBodyChunk(StringRef BodyChunk) {
  if (!ResponseBuffer.Body)
    return createStringError(errc::io_error,
                             "Unallocated response buffer. HTTP Body data "
                             "received before Content-Length header.");
  if (Offset + BodyChunk.size() > ResponseBuffer.Body->getBufferSize())
    return createStringError(errc::io_error,
                             "Content size exceeds buffer size.");
  memcpy(ResponseBuffer.Body->getBufferStart() + Offset, BodyChunk.data(),
         BodyChunk.size());
  Offset += BodyChunk.size();
  return Error::success();
}

Error BufferedHTTPResponseHandler::handleStatusCode(unsigned Code) {
  ResponseBuffer.Code = Code;
  return Error::success();
}

Expected<HTTPResponseBuffer> HTTPClient::perform(const HTTPRequest &Request) {
  BufferedHTTPResponseHandler Handler;
  if (Error Err = perform(Request, Handler))
    return std::move(Err);
  return std::move(Handler.ResponseBuffer);
}

Expected<HTTPResponseBuffer> HTTPClient::get(StringRef Url) {
  HTTPRequest Request(Url);
  return perform(Request);
}

HTTPClient::HTTPClient() = default;

HTTPClient::~HTTPClient() = default;

bool HTTPClient::isAvailable() { return false; }

void HTTPClient::cleanup() {}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {}

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  llvm_unreachable("No HTTP Client implementation available.");
}
