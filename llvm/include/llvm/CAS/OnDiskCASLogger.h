//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares interface for OnDiskCASLogger, an interface that can be
/// used to log CAS events to help debugging CAS errors.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKLOGGER_H
#define LLVM_CAS_ONDISKLOGGER_H

#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
class raw_fd_ostream;
class Twine;
} // namespace llvm

namespace llvm::cas::ondisk {

/// Interface for logging low-level on-disk cas operations.
///
/// This log is intended to mirror the low-level details of the CAS in order to
/// aid with debugging corruption or other issues with the on-disk format.
class OnDiskCASLogger {
public:
  /// Create or append to a log file inside the given CAS directory \p Path.
  ///
  /// \param Path The parent directory of the log file.
  /// \param LogAllocations Whether to log all low-level allocations. This is
  ///                       on the order of twice as expensive to log.
  static Expected<std::unique_ptr<OnDiskCASLogger>> open(const Twine &Path,
                                                         bool LogAllocations);

  /// Create or append to a log file inside the given CAS directory \p Path if
  /// logging is enabled by the environment variable \c LLVM_CAS_LOG. If
  /// LLVM_CAS_LOG is set >= 2 then also log allocations.
  static Expected<std::unique_ptr<OnDiskCASLogger>>
  openIfEnabled(const Twine &Path);

  ~OnDiskCASLogger();

  /// An offset into an \c OnDiskTrieRawHashMap.
  using TrieOffset = int64_t;

  void log_compare_exchange_strong(void *Region, TrieOffset Trie, size_t SlotI,
                                   TrieOffset Expected, TrieOffset New,
                                   TrieOffset Previous);
  void log_SubtrieHandle_create(void *Region, TrieOffset Trie,
                                uint32_t StartBit, uint32_t NumBits);
  void log_HashMappedTrieHandle_createRecord(void *Region,
                                             TrieOffset TrieOffset,
                                             ArrayRef<uint8_t> Hash);
  void log_MappedFileRegionArena_resizeFile(StringRef Path, size_t Before,
                                            size_t After);
  void log_MappedFileRegionArena_create(StringRef Path, int FD, void *Region,
                                        size_t Capacity, size_t Size);
  void log_MappedFileRegionArena_oom(StringRef Path, size_t Capacity,
                                     size_t Size, size_t AllocSize);
  void log_MappedFileRegionArena_close(StringRef Path);
  void log_MappedFileRegionArena_allocate(void *Region, TrieOffset Off,
                                          size_t Size);
  void log_UnifiedOnDiskCache_collectGarbage(StringRef Path);
  void log_UnifiedOnDiskCache_validateIfNeeded(
      StringRef Path, uint64_t BootTime, uint64_t ValidationTime,
      bool CheckHash, bool AllowRecovery, bool Force,
      std::optional<StringRef> LLVMCas, StringRef ValidationError, bool Skipped,
      bool Recovered);
  void log_TempFile_create(StringRef Name);
  void log_TempFile_keep(StringRef TmpName, StringRef Name, std::error_code EC);
  void log_TempFile_remove(StringRef TmpName, std::error_code EC);

private:
  OnDiskCASLogger(raw_fd_ostream &OS, bool LogAllocations);

  raw_fd_ostream &OS;
  bool LogAllocations;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_CAS_ONDISKLOGGER_H
