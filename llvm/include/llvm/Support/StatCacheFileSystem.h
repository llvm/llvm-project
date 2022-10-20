//===- StatCacheFileSystem.h - Status Caching Proxy File System -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STATCACHEFILESYSTEM_H
#define LLVM_SUPPORT_STATCACHEFILESYSTEM_H

#include "llvm/Support/VirtualFileSystem.h"

#include <list>

namespace llvm {
template <typename T> class OnDiskIterableChainedHashTable;
template <typename T> class OnDiskChainedHashTableGenerator;

namespace vfs {

/// A ProxyFileSystem using cached information for status() rather than going to
/// the underlying filesystem.
///
/// When dealing with a huge tree of (mostly) immutable filesystem content
/// like an SDK, it can be very costly to ask the underlying filesystem for
/// `stat` data. Even when caching the `stat`s internally, having many
/// concurrent Clangs accessing the same tree in a similar way causes
/// contention. As SDK files are mostly immutable, we can pre-compute the status
/// information using clang-stat-cache and use that information directly without
/// accessing the real filesystem until Clang needs to open a file. This can
/// speed up module verification and HeaderSearch by significant amounts.
class StatCacheFileSystem : public llvm::vfs::ProxyFileSystem {
  class StatCacheLookupInfo;
  using StatCacheType =
      llvm::OnDiskIterableChainedHashTable<StatCacheLookupInfo>;

  class StatCacheGenerationInfo;
  using StatCacheGeneratorType =
      llvm::OnDiskChainedHashTableGenerator<StatCacheGenerationInfo>;

  explicit StatCacheFileSystem(std::unique_ptr<llvm::MemoryBuffer> CacheFile,
                               IntrusiveRefCntPtr<FileSystem> FS,
                               bool IsCaseSensitive);

public:
  /// Create a StatCacheFileSystem from the passed \a CacheBuffer, a
  /// MemoryBuffer representing the contents of the \a CacheFilename file. The
  /// returned filesystem will be overlaid on top of \a FS.
  static Expected<IntrusiveRefCntPtr<StatCacheFileSystem>>
  create(std::unique_ptr<llvm::MemoryBuffer> CacheBuffer,
         IntrusiveRefCntPtr<FileSystem> FS);

  /// The status override which will consult the cache if \a Path is in the
  /// cached filesystem tree.
  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override;

public:
  /// A helper class to generate stat caches.
  class StatCacheWriter {
    llvm::SmallString<128> BaseDir;
    bool IsCaseSensitive;
    uint64_t ValidityToken;
    StatCacheGeneratorType *Generator;
    std::list<std::string> PathStorage;

  public:
    /// Create a StatCacheWriter
    ///
    /// \param BaseDir The base directory for the path. Every filename passed to
    ///                addEntry() needs to start with this base directory.
    /// \param Status The status entry for the base directory.
    /// \param IsCaseSensitive Whether the cache is case sensitive.
    /// \param ValidityToken A 64 bits token that gets embedded in the cache and
    ///                      can be used by generator tools to check for the
    ///                      cache validity in a platform-specific way.
    StatCacheWriter(StringRef BaseDir, const sys::fs::file_status &Status,
                    bool IsCaseSensitive, uint64_t ValidityToken = 0);
    ~StatCacheWriter();

    /// Add a cache entry storing \a Status for the file at \a Path.
    void addEntry(StringRef Path, const sys::fs::file_status &Status);

    /// Write the cache file to \a Out.
    size_t writeStatCache(raw_fd_ostream &Out);
  };

public:
  /// Validate that the file content in \a Buffer is a valid stat cache file.
  /// \a BaseDir, \a IsCaseSensitive and \a ValidityToken are output parameters
  /// that get populated by this call.
  static Error validateCacheFile(llvm::MemoryBufferRef Buffer,
                                 StringRef &BaseDir, bool &IsCaseSensitive,
                                 bool &VersionMatch, uint64_t &ValidityToken);

  /// Update the ValidityToken data in \a CacheFile.
  static void updateValidityToken(raw_fd_ostream &CacheFile,
                                  uint64_t ValidityToken);

private:
  std::unique_ptr<llvm::MemoryBuffer> StatCacheFile;
  llvm::StringRef StatCachePrefix;
  std::unique_ptr<StatCacheType> StatCache;
  bool IsCaseSensitive = true;
};

} // namespace vfs
} // namespace llvm

#endif // LLVM_SUPPORT_STATCACHEFILESYSTEM_H
