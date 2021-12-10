//===-- llvm/Support/HTTPClient.h - HTTP client library ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the HTTPClient, HTTPMethod,
/// HTTPResponseHandler, and BufferedHTTPResponseHandler classes, as well as
/// the HTTPResponseBuffer and HTTPRequest structs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HTTP_CLIENT_H
#define LLVM_SUPPORT_HTTP_CLIENT_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

enum class HTTPMethod { GET };

/// A stateless description of an outbound HTTP request.
struct HTTPRequest {
  SmallString<128> Url;
  HTTPMethod Method = HTTPMethod::GET;
  bool FollowRedirects = true;
  HTTPRequest(StringRef Url);
};

bool operator==(const HTTPRequest &A, const HTTPRequest &B);

/// A handler for state updates occurring while an HTTPRequest is performed.
/// Can trigger the client to abort the request by returning an Error from any
/// of its methods.
class HTTPResponseHandler {
public:
  /// Processes one line of HTTP response headers.
  virtual Error handleHeaderLine(StringRef HeaderLine) = 0;

  /// Processes an additional chunk of bytes of the HTTP response body.
  virtual Error handleBodyChunk(StringRef BodyChunk) = 0;

  /// Processes the HTTP response status code.
  virtual Error handleStatusCode(unsigned Code) = 0;

protected:
  ~HTTPResponseHandler();
};

/// An HTTP response status code bundled with a buffer to store the body.
struct HTTPResponseBuffer {
  unsigned Code = 0;
  std::unique_ptr<WritableMemoryBuffer> Body;
};

/// A simple handler which writes returned data to an HTTPResponseBuffer.
/// Ignores all headers except the Content-Length, which it uses to
/// allocate an appropriately-sized Body buffer.
class BufferedHTTPResponseHandler final : public HTTPResponseHandler {
  size_t Offset = 0;

public:
  /// Stores the data received from the HTTP server.
  HTTPResponseBuffer ResponseBuffer;

  /// These callbacks store the body and status code in an HTTPResponseBuffer
  /// allocated based on Content-Length. The Content-Length header must be
  /// handled by handleHeaderLine before any calls to handleBodyChunk.
  Error handleHeaderLine(StringRef HeaderLine) override;
  Error handleBodyChunk(StringRef BodyChunk) override;
  Error handleStatusCode(unsigned Code) override;
};

/// A reusable client that can perform HTTPRequests through a network socket.
class HTTPClient {
#ifdef LLVM_ENABLE_CURL
  void *Curl = nullptr;
#endif

public:
  HTTPClient();
  ~HTTPClient();

  static bool IsInitialized;

  /// Returns true only if LLVM has been compiled with a working HTTPClient.
  static bool isAvailable();

  /// Must be called at the beginning of a program, while it is a single thread.
  static void initialize();

  /// Must be called at the end of a program, while it is a single thread.
  static void cleanup();

  /// Sets the timeout for the entire request, in milliseconds. A zero or
  /// negative value means the request never times out.
  void setTimeout(std::chrono::milliseconds Timeout);

  /// Performs the Request, passing response data to the Handler. Returns all
  /// errors which occur during the request. Aborts if an error is returned by a
  /// Handler method.
  Error perform(const HTTPRequest &Request, HTTPResponseHandler &Handler);

  /// Performs the Request with the default BufferedHTTPResponseHandler, and
  /// returns its HTTPResponseBuffer or an Error.
  Expected<HTTPResponseBuffer> perform(const HTTPRequest &Request);

  /// Performs an HTTPRequest with the default configuration to make a GET
  /// request to the given Url. Returns an HTTPResponseBuffer or an Error.
  Expected<HTTPResponseBuffer> get(StringRef Url);
};

} // end namespace llvm

#endif // LLVM_SUPPORT_HTTP_CLIENT_H
