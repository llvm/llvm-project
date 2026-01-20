//===- Caching.h - LLVM Local File Cache ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CachedFileStream and the localCache function, which
// simplifies caching files on the local filesystem in a directory whose
// contents are managed by a CachePruningPolicy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CACHING_H
#define LLVM_SUPPORT_CACHING_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

/// This class wraps an output stream for a file. Most clients should just be
/// able to return an instance of this base class from the stream callback, but
/// if a client needs to perform some action after the stream is written to,
/// that can be done by deriving from this class and overriding the destructor
/// or the commit() method.
class CachedFileStream {
public:
  CachedFileStream(std::unique_ptr<raw_pwrite_stream> OS,
                   std::string OSPath = "")
      : OS(std::move(OS)), ObjectPathName(OSPath) {}

  /// Must be called exactly once after the writes to OS have been completed
  /// but before the CachedFileStream object is destroyed.
  virtual Error commit() {
    if (Committed)
      return createStringError(make_error_code(std::errc::invalid_argument),
                               Twine("CacheStream already committed."));
    Committed = true;

    return Error::success();
  }

  bool Committed = false;
  std::unique_ptr<raw_pwrite_stream> OS;
  std::string ObjectPathName;
  virtual ~CachedFileStream() {
    if (!Committed)
      report_fatal_error("CachedFileStream was not committed.\n");
  }
};

/// This type defines the callback to add a file that is generated on the fly.
///
/// Stream callbacks must be thread safe.
using AddStreamFn = std::function<Expected<std::unique_ptr<CachedFileStream>>(
    unsigned Task, const Twine &ModuleName)>;

/// This is a callable that manages file caching operations. It accepts a task
/// ID \p Task, a unique key \p Key, and a module name \p ModuleName, and
/// returns AddStreamFn(). This function determines whether a cache hit or miss
/// occurs and handles the appropriate actions.
using FileCacheFunction = std::function<Expected<AddStreamFn>(
    unsigned Task, StringRef Key, const Twine &ModuleName)>;

/// This type represents a file cache system that manages caching of files.
/// It encapsulates a caching function and the directory path where the cache is
/// stored. To request an item from the cache, pass a unique string as the Key.
/// For hits, the cached file will be added to the link and this function will
/// return AddStreamFn(). For misses, the cache will return a stream callback
/// which must be called at most once to produce content for the stream. The
/// file stream produced by the stream callback will add the file to the link
/// after the stream is written to. ModuleName is the unique module identifier
/// for the bitcode module the cache is being checked for.
///
/// Clients generally look like this:
///
/// if (AddStreamFn AddStream = Cache(Task, Key, ModuleName))
///   ProduceContent(AddStream);
///
/// CacheDirectoryPath stores the directory path where cached files are kept.
struct FileCache {
  FileCache(FileCacheFunction CacheFn, const std::string &DirectoryPath)
      : CacheFunction(std::move(CacheFn)), CacheDirectoryPath(DirectoryPath) {}
  FileCache() = default;

  Expected<AddStreamFn> operator()(unsigned Task, StringRef Key,
                                   const Twine &ModuleName) {
    assert(isValid() && "Invalid cache function");
    return CacheFunction(Task, Key, ModuleName);
  }
  const std::string &getCacheDirectoryPath() const {
    return CacheDirectoryPath;
  }
  bool isValid() const { return static_cast<bool>(CacheFunction); }

private:
  FileCacheFunction CacheFunction = nullptr;
  std::string CacheDirectoryPath;
};

/// This type defines the callback to add a pre-existing file (e.g. in a cache).
///
/// Buffer callbacks must be thread safe.
using AddBufferFn = std::function<void(unsigned Task, const Twine &ModuleName,
                                       std::unique_ptr<MemoryBuffer> MB)>;

/// Create a local file system cache which uses the given cache name, temporary
/// file prefix, cache directory and file callback.  This function does not
/// immediately create the cache directory if it does not yet exist; this is
/// done lazily the first time a file is added.  The cache name appears in error
/// messages for errors during caching. The temporary file prefix is used in the
/// temporary file naming scheme used when writing files atomically.
LLVM_ABI Expected<FileCache> localCache(
    const Twine &CacheNameRef, const Twine &TempFilePrefixRef,
    const Twine &CacheDirectoryPathRef,
    AddBufferFn AddBuffer = [](size_t Task, const Twine &ModuleName,
                               std::unique_ptr<MemoryBuffer> MB) {});
} // namespace llvm

#endif
