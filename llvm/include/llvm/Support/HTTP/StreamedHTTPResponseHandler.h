//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// An HTTPResponseHandler that streams the response body to a CachedFileStream.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HTTP_STREAMEDHTTPRESPONSEHANDLER_H
#define LLVM_SUPPORT_HTTP_STREAMEDHTTPRESPONSEHANDLER_H

#include "llvm/Support/Caching.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/HTTP/HTTPClient.h"
#include <functional>
#include <memory>

namespace llvm {

/// A handler which streams the returned data to a CachedFileStream. The cache
/// file is only created if a 200 OK status is observed.
class StreamedHTTPResponseHandler : public HTTPResponseHandler {
  using CreateStreamFn =
      std::function<Expected<std::unique_ptr<CachedFileStream>>()>;
  CreateStreamFn CreateStream;
  HTTPClient &Client;
  std::unique_ptr<CachedFileStream> FileStream;

public:
  StreamedHTTPResponseHandler(CreateStreamFn CreateStream, HTTPClient &Client)
      : CreateStream(std::move(CreateStream)), Client(Client) {}

  /// Must be called exactly once after the writes have been completed
  /// but before the StreamedHTTPResponseHandler object is destroyed.
  Error commit();

  virtual ~StreamedHTTPResponseHandler() = default;

  Error handleBodyChunk(StringRef BodyChunk) override;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_HTTP_STREAMEDHTTPRESPONSEHANDLER_H
