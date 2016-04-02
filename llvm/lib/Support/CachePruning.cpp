//===-CachePruning.cpp - LLVM Cache Directory Pruning ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pruning of a directory based on least recently used.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CachePruning.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

using namespace llvm;

/// Write a new timestamp file with the given path. This is used for the pruning
/// interval option.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  raw_fd_ostream Out(TimestampFile.str(), EC, sys::fs::F_None);
}

/// Prune the cache of files that haven't been accessed in a long time.
bool CachePruning::prune() {
  SmallString<128> TimestampFile(Path);
  sys::path::append(TimestampFile, "llvmcache.timestamp");

  if (Expiration == 0 && PercentageOfAvailableSpace == 0)
    // Nothing will be pruned, early exit
    return false;

  // Try to stat() the timestamp file.
  sys::fs::file_status FileStatus;
  sys::TimeValue CurrentTime = sys::TimeValue::now();
  if (sys::fs::status(TimestampFile, FileStatus)) {
    if (errno == ENOENT) {
      // If the timestamp file wasn't there, create one now.
      writeTimestampFile(TimestampFile);
    } else {
      // Unknown error?
      return false;
    }
  } else {
    if (Interval) {
      // Check whether the time stamp is older than our pruning interval.
      // If not, do nothing.
      sys::TimeValue TimeStampModTime = FileStatus.getLastModificationTime();
      auto TimeInterval = sys::TimeValue(sys::TimeValue::SecondsType(Interval));
      if (CurrentTime - TimeStampModTime <= TimeInterval)
        return false;
    }
    // Write a new timestamp file so that nobody else attempts to prune.
    // There is a benign race condition here, if two processes happen to
    // notice at the same time that the timestamp is out-of-date.
    writeTimestampFile(TimestampFile);
  }

  bool ShouldComputeSize = (PercentageOfAvailableSpace > 0);

  // Keep track of space
  std::set<std::pair<uint64_t, std::string>> FileSizes;
  uint64_t TotalSize = 0;
  // Helper to add a path to the set of files to consider for size-based
  // pruning, sorted by last accessed time.
  auto AddToFileListForSizePruning =
      [&](StringRef Path, sys::TimeValue FileAccessTime) {
        if (!ShouldComputeSize)
          return;
        TotalSize += FileStatus.getSize();
        FileSizes.insert(
            std::make_pair(FileAccessTime.seconds(), std::string(Path)));
      };

  // Walk the entire directory cache, looking for unused files.
  std::error_code EC;
  SmallString<128> CachePathNative;
  sys::path::native(Path, CachePathNative);
  auto TimeExpiration = sys::TimeValue(sys::TimeValue::SecondsType(Expiration));
  // Walk all of the files within this directory.
  for (sys::fs::directory_iterator File(CachePathNative, EC), FileEnd;
       File != FileEnd && !EC; File.increment(EC)) {
    // Do not touch the timestamp.
    if (File->path() == TimestampFile)
      continue;

    // Look at this file. If we can't stat it, there's nothing interesting
    // there.
    if (sys::fs::status(File->path(), FileStatus))
      continue;

    // If the file hasn't been used recently enough, delete it
    sys::TimeValue FileAccessTime = FileStatus.getLastAccessedTime();
    if (CurrentTime - FileAccessTime > TimeExpiration) {
      sys::fs::remove(File->path());
      continue;
    }

    // Leave it here for now, but add it to the list of size-based pruning.
    AddToFileListForSizePruning(File->path(), FileAccessTime);
  }

  // Prune for size now if needed
  if (ShouldComputeSize) {
    auto ErrOrSpaceInfo = sys::fs::disk_space(Path);
    if (!ErrOrSpaceInfo) {
      report_fatal_error("Can't get available size");
    }
    sys::fs::space_info SpaceInfo = ErrOrSpaceInfo.get();
    auto AvailableSpace = TotalSize + SpaceInfo.free;
    auto FileAndSize = FileSizes.rbegin();
    // Remove the oldest accessed files first, till we get below the threshold
    while (((100 * TotalSize) / AvailableSpace) > PercentageOfAvailableSpace &&
           FileAndSize != FileSizes.rend()) {
      // Remove the file.
      sys::fs::remove(FileAndSize->second);
      // Update size
      TotalSize -= FileAndSize->first;
      ++FileAndSize;
    }
  }
  return true;
}
