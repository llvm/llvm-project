//===-- llvm/Debuginfod/HTTPClient.cpp - HTTP client library ----*- C++ -*-===//
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

#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#ifdef LLVM_ENABLE_CURL
#include <curl/curl.h>
#endif

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

bool HTTPClient::IsInitialized = false;

class HTTPClientCleanup {
public:
  ~HTTPClientCleanup() { HTTPClient::cleanup(); }
};
static const HTTPClientCleanup Cleanup;

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

#ifdef LLVM_ENABLE_CURL

bool HTTPClient::isAvailable() { return true; }

void HTTPClient::initialize() {
  if (!IsInitialized) {
    curl_global_init(CURL_GLOBAL_ALL);
    IsInitialized = true;
  }
}

void HTTPClient::cleanup() {
  if (IsInitialized) {
    curl_global_cleanup();
    IsInitialized = false;
  }
}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {
  if (Timeout < std::chrono::milliseconds(0))
    Timeout = std::chrono::milliseconds(0);
  curl_easy_setopt(Curl, CURLOPT_TIMEOUT_MS, Timeout.count());
}

/// CurlHTTPRequest and the curl{Header,Write}Function are implementation
/// details used to work with Curl. Curl makes callbacks with a single
/// customizable pointer parameter.
struct CurlHTTPRequest {
  CurlHTTPRequest(HTTPResponseHandler &Handler) : Handler(Handler) {}
  void storeError(Error Err) {
    ErrorState = joinErrors(std::move(Err), std::move(ErrorState));
  }
  HTTPResponseHandler &Handler;
  llvm::Error ErrorState = Error::success();
};

static size_t curlHeaderFunction(char *Contents, size_t Size, size_t NMemb,
                                 CurlHTTPRequest *CurlRequest) {
  assert(Size == 1 && "The Size passed by libCURL to CURLOPT_HEADERFUNCTION "
                      "should always be 1.");
  if (Error Err =
          CurlRequest->Handler.handleHeaderLine(StringRef(Contents, NMemb))) {
    CurlRequest->storeError(std::move(Err));
    return 0;
  }
  return NMemb;
}

static size_t curlWriteFunction(char *Contents, size_t Size, size_t NMemb,
                                CurlHTTPRequest *CurlRequest) {
  Size *= NMemb;
  if (Error Err =
          CurlRequest->Handler.handleBodyChunk(StringRef(Contents, Size))) {
    CurlRequest->storeError(std::move(Err));
    return 0;
  }
  return Size;
}

HTTPClient::HTTPClient() {
  assert(IsInitialized &&
         "Must call HTTPClient::initialize() at the beginning of main().");
  if (Curl)
    return;
  assert((Curl = curl_easy_init()) && "Curl could not be initialized.");
  // Set the callback hooks.
  curl_easy_setopt(Curl, CURLOPT_WRITEFUNCTION, curlWriteFunction);
  curl_easy_setopt(Curl, CURLOPT_HEADERFUNCTION, curlHeaderFunction);
}

HTTPClient::~HTTPClient() { curl_easy_cleanup(Curl); }

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  if (Request.Method != HTTPMethod::GET)
    return createStringError(errc::invalid_argument,
                             "Unsupported CURL request method.");

  SmallString<128> Url = Request.Url;
  curl_easy_setopt(Curl, CURLOPT_URL, Url.c_str());
  curl_easy_setopt(Curl, CURLOPT_FOLLOWLOCATION, Request.FollowRedirects);

  CurlHTTPRequest CurlRequest(Handler);
  curl_easy_setopt(Curl, CURLOPT_WRITEDATA, &CurlRequest);
  curl_easy_setopt(Curl, CURLOPT_HEADERDATA, &CurlRequest);
  CURLcode CurlRes = curl_easy_perform(Curl);
  if (CurlRes != CURLE_OK)
    return joinErrors(std::move(CurlRequest.ErrorState),
                      createStringError(errc::io_error,
                                        "curl_easy_perform() failed: %s\n",
                                        curl_easy_strerror(CurlRes)));
  if (CurlRequest.ErrorState)
    return std::move(CurlRequest.ErrorState);

  unsigned Code;
  curl_easy_getinfo(Curl, CURLINFO_RESPONSE_CODE, &Code);
  if (Error Err = Handler.handleStatusCode(Code))
    return joinErrors(std::move(CurlRequest.ErrorState), std::move(Err));

  return std::move(CurlRequest.ErrorState);
}

#else

HTTPClient::HTTPClient() = default;

HTTPClient::~HTTPClient() = default;

bool HTTPClient::isAvailable() { return false; }

void HTTPClient::initialize() {}

void HTTPClient::cleanup() {}

void HTTPClient::setTimeout(std::chrono::milliseconds Timeout) {}

Error HTTPClient::perform(const HTTPRequest &Request,
                          HTTPResponseHandler &Handler) {
  llvm_unreachable("No HTTP Client implementation available.");
}

#endif
