//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/HTTP/StreamedHTTPResponseHandler.h"

namespace llvm {

Error StreamedHTTPResponseHandler::handleBodyChunk(StringRef BodyChunk) {
  if (!FileStream) {
    unsigned Code = Client.responseCode();
    if (Code && Code != 200)
      return Error::success();
    Expected<std::unique_ptr<CachedFileStream>> FileStreamOrError =
        CreateStream();
    if (!FileStreamOrError)
      return FileStreamOrError.takeError();
    FileStream = std::move(*FileStreamOrError);
  }
  *FileStream->OS << BodyChunk;
  return Error::success();
}

Error StreamedHTTPResponseHandler::commit() {
  if (FileStream)
    return FileStream->commit();
  return Error::success();
}

} // namespace llvm
