//===-- llvm/RemoteCachingService/LLVMCASCacheProvider.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_LLVMCASCACHEPROVIDER_H
#define LLVM_REMOTECACHINGSERVICE_LLVMCASCACHEPROVIDER_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm::cas {
class ActionCache;
class ObjectStore;

namespace remote {
class RemoteCacheProvider;

/// Returns a \p RemoteCacheProvider that is implemented using an on-disk \p
/// cas::ObjectStore and \p cas::ActionCache.
std::unique_ptr<RemoteCacheProvider>
createLLVMCASCacheProvider(StringRef TempPath, std::unique_ptr<ObjectStore> CAS,
                           std::unique_ptr<ActionCache> Cache);

} // namespace remote
} // namespace llvm::cas

#endif
