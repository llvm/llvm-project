//===- Caching.h - LLVM Link Time Optimizer Configuration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the localCache function, which allows clients to add a
// filesystem cache to ThinLTO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_CACHING_H
#define LLVM_LTO_CACHING_H

#include "llvm/LTO/LTO.h"

namespace llvm {
namespace lto {

/// This type defines the callback to add a pre-existing native object file
/// (e.g. in a cache).
///
/// Buffer callbacks must be thread safe.
using AddBufferFn =
    std::function<void(unsigned Task, std::unique_ptr<MemoryBuffer> MB)>;

/// Create a local file system cache which uses the given cache directory and
/// file callback. This function also creates the cache directory if it does not
/// already exist.
Expected<NativeObjectCache> localCache(StringRef CacheDirectoryPath,
                                       AddBufferFn AddBuffer);

} // namespace lto
} // namespace llvm

#endif
