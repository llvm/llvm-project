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

#include "llvm/Support/Compiler.h"
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
  LLVM_ABI static Expected<std::unique_ptr<OnDiskCASLogger>>
  open(const Twine &Path, bool LogAllocations);

  /// Create or append to a log file inside the given CAS directory \p Path if
  /// logging is enabled by the environment variable \c LLVM_CAS_LOG. If
  /// LLVM_CAS_LOG is set >= 2 then also log allocations.
  LLVM_ABI static Expected<std::unique_ptr<OnDiskCASLogger>>
  openIfEnabled(const Twine &Path);

  LLVM_ABI ~OnDiskCASLogger();

  /// An offset into an \c OnDiskTrieRawHashMap.
  using TrieOffset = int64_t;

  LLVM_ABI void logSubtrieHandleCmpXchg(void *Region, TrieOffset Trie,
                                        size_t SlotI, TrieOffset Expected,
                                        TrieOffset New, TrieOffset Previous);
  LLVM_ABI void logSubtrieHandleCreate(void *Region, TrieOffset Trie,
                                       uint32_t StartBit, uint32_t NumBits);
  LLVM_ABI void logHashMappedTrieHandleCreateRecord(void *Region,
                                                    TrieOffset TrieOffset,
                                                    ArrayRef<uint8_t> Hash);
  LLVM_ABI void logMappedFileRegionArenaResizeFile(StringRef Path,
                                                   size_t Before, size_t After);
  LLVM_ABI void logMappedFileRegionArenaCreate(StringRef Path, int FD,
                                               void *Region, size_t Capacity,
                                               size_t Size);
  LLVM_ABI void logMappedFileRegionArenaOom(StringRef Path, size_t Capacity,
                                            size_t Size, size_t AllocSize);
  LLVM_ABI void logMappedFileRegionArenaClose(StringRef Path);
  LLVM_ABI void logMappedFileRegionArenaAllocate(void *Region, TrieOffset Off,
                                                 size_t Size);
  LLVM_ABI void logUnifiedOnDiskCacheCollectGarbage(StringRef Path);
  LLVM_ABI void logUnifiedOnDiskCacheValidateIfNeeded(
      StringRef Path, uint64_t BootTime, uint64_t ValidationTime,
      bool CheckHash, bool AllowRecovery, bool Force,
      std::optional<StringRef> LLVMCas, StringRef ValidationError, bool Skipped,
      bool Recovered);
  LLVM_ABI void logTempFileCreate(StringRef Name);
  LLVM_ABI void logTempFileKeep(StringRef TmpName, StringRef Name,
                                std::error_code EC);
  LLVM_ABI void logTempFileRemove(StringRef TmpName, std::error_code EC);

private:
  OnDiskCASLogger(raw_fd_ostream &OS, bool LogAllocations);

  raw_fd_ostream &OS;
  bool LogAllocations;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_CAS_ONDISKLOGGER_H
