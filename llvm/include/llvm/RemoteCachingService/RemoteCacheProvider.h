//===-- llvm/RemoteCachingService/RemoteCacheProvider.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_REMOTECACHEPROVIDER_H
#define LLVM_REMOTECACHINGSERVICE_REMOTECACHEPROVIDER_H

#include "llvm/Support/Error.h"

namespace llvm::cas::remote {

/// A caching backend that provides the underlying functionality for
/// implementing the remote cache protocol.
///
/// While the functions are running they are blocking \p RemoteCacheServer from
/// serving more requests, implementations should do the work asynchronously.
class RemoteCacheProvider {
public:
  virtual ~RemoteCacheProvider() = default;

  virtual void GetValueAsync(
      std::string Key,
      std::function<void(Expected<std::optional<std::string>>)> Receiver) = 0;
  virtual void PutValueAsync(std::string Key, std::string Value,
                             std::function<void(Error)> Receiver) = 0;

  struct BlobContents {
    bool IsFilePath = false;
    std::string DataOrPath;
  };

  struct LoadResponse {
    bool KeyNotFound = false;
    BlobContents Blob;
  };

  virtual void
  CASLoadAsync(std::string CASID, bool WriteToDisk,
               std::function<void(Expected<LoadResponse>)> Receiver) = 0;
  virtual void
  CASSaveAsync(BlobContents Blob,
               std::function<void(Expected<std::string>)> Receiver) = 0;

  struct GetResponse {
    bool KeyNotFound = false;
    BlobContents Blob;
    SmallVector<std::string> Refs;
  };

  virtual void
  CASGetAsync(std::string CASID, bool WriteToDisk,
              std::function<void(Expected<GetResponse>)> Receiver) = 0;
  virtual void
  CASPutAsync(BlobContents Blob, SmallVector<std::string> Refs,
              std::function<void(Expected<std::string>)> Receiver) = 0;

private:
  virtual void anchor();
};

} // namespace llvm::cas::remote

#endif
