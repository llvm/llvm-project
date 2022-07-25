//===-- RemoteCacheProvider.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_REMOTECACHEPROVIDER_H
#define LLVM_REMOTECACHINGSERVICE_REMOTECACHEPROVIDER_H

#include "compilation_caching_cas.grpc.pb.h"
#include "compilation_caching_kv.grpc.pb.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm::cas {

class ActionCache;
class ObjectStore;

namespace remote {

/// A caching backend that provides the underlying functionality for
/// implementing the remote cache protocol.
///
/// While the functions are running they are blocking \p RemoteCacheServer from
/// serving more requests, implementations should do the work asynchronously.
class RemoteCacheProvider {
public:
  virtual ~RemoteCacheProvider() = default;

  virtual void GetValueAsync(
      const compilation_cache_service::keyvalue::v1::GetValueRequest &Request,
      std::function<void(
          const compilation_cache_service::keyvalue::v1::GetValueResponse &)>
          Receiver) = 0;
  virtual void PutValueAsync(
      const compilation_cache_service::keyvalue::v1::PutValueRequest &Request,
      std::function<void(
          const compilation_cache_service::keyvalue::v1::PutValueResponse &)>
          Receiver) = 0;

  virtual void CASLoadAsync(
      const compilation_cache_service::cas::v1::CASLoadRequest &Request,
      std::function<
          void(const compilation_cache_service::cas::v1::CASLoadResponse &)>
          Receiver) = 0;
  virtual void CASSaveAsync(
      const compilation_cache_service::cas::v1::CASSaveRequest &Request,
      std::function<
          void(const compilation_cache_service::cas::v1::CASSaveResponse &)>
          Receiver) = 0;
  virtual void CASGetAsync(
      const compilation_cache_service::cas::v1::CASGetRequest &Request,
      std::function<
          void(const compilation_cache_service::cas::v1::CASGetResponse &)>
          Receiver) = 0;
  virtual void CASPutAsync(
      const compilation_cache_service::cas::v1::CASPutRequest &Request,
      std::function<
          void(const compilation_cache_service::cas::v1::CASPutResponse &)>
          Receiver) = 0;
};

/// Returns a \p RemoteCacheProvider that is implemented using an on-disk \p
/// cas::ObjectStore and \p cas::ActionCache.
std::unique_ptr<RemoteCacheProvider>
createLLVMCASCacheProvider(StringRef TempPath, std::unique_ptr<ObjectStore> CAS,
                           std::unique_ptr<ActionCache> Cache);

} // namespace remote
} // namespace llvm::cas

#endif
