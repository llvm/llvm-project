//===- Caching.h - LLVM Link Time Optimizer Configuration -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the lto::CacheObjectOutput data structure, which allows
// clients to add a filesystem cache to ThinLTO
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_CACHING_H
#define LLVM_LTO_CACHING_H

#include "llvm/ADT/SmallString.h"
#include "llvm/LTO/Config.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace lto {
/// Type for client-supplied callback when a buffer is loaded from the cache.
typedef std::function<void(std::string)> AddBufferFn;

/// Manage caching on the filesystem.
///
/// The general scheme is the following:
///
/// void do_stuff(AddBufferFn CallBack) {
///   /* ... */
///   {
///     /* Create the CacheObjectOutput pointing to a cache directory */
///     auto Output = CacheObjectOutput("/tmp/cache", CallBack)
///
///     /* Call some processing function */
///     process(Output);
///
///   } /* Callback is only called now, on destruction of the Output object */
///   /* ... */
/// }
///
///
/// void process(NativeObjectOutput &Output) {
///   /* check if caching is supported */
///   if (Output.isCachingEnabled()) {
///     auto Key = ComputeKeyForEntry(...); // "expensive" call
///     if (Output.tryLoadFromCache())
///        return; // Cache hit
///   }
///
///   auto OS = Output.getStream();
///
///   OS << ...;
///   /* Note that the callback is not called here, but only when the caller
///      destroys Output */
/// }
///
class CacheObjectOutput : public NativeObjectOutput {
  /// Path to the on-disk cache directory
  StringRef CacheDirectoryPath;
  /// Path to this entry in the cache, initialized by tryLoadFromCache().
  SmallString<128> EntryPath;
  /// Path to temporary file used to buffer output that will be committed to the
  /// cache entry when this object is destroyed
  SmallString<128> TempFilename;
  /// User-supplied callback, used to provide path to cache entry
  /// (potentially after creating it).
  AddBufferFn AddBuffer;

public:
  /// The destructor pulls the entry from the cache and calls the AddBuffer
  /// callback, after committing the entry into the cache on miss.
  ~CacheObjectOutput();

  /// Create a CacheObjectOutput: the client is supposed to create it in the
  /// callback supplied to LTO::run. The \p CacheDirectoryPath points to the
  /// directory on disk where to store the cache, and \p AddBuffer will be
  /// called when the buffer is ready to be pulled out of the cache
  /// (potentially after creating it).
  CacheObjectOutput(StringRef CacheDirectoryPath, AddBufferFn AddBuffer)
      : CacheDirectoryPath(CacheDirectoryPath), AddBuffer(AddBuffer) {}

  /// Return an allocated stream for the output, or null in case of failure.
  std::unique_ptr<raw_pwrite_stream> getStream() override;

  /// Set EntryPath, try loading from a possible cache first, return true on
  /// cache hit.
  bool tryLoadFromCache(StringRef Key) override;

  /// Returns true to signal that this implementation of NativeObjectFile
  /// support caching.
  bool isCachingEnabled() const override { return true; }
};

} // namespace lto
} // namespace llvm

#endif
