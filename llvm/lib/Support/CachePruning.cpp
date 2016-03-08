//===-CachePruning.cpp - LLVM Cache Director Pruning ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Thin Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CachePruning.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/stat.h>
#include <sys/param.h>
#include <sys/mount.h>

#include <set>

using namespace llvm;

/// \brief Write a new timestamp file with the given path.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(TimestampFile.str(), EC, llvm::sys::fs::F_None);
}

/// \brief Prune the cache of files that haven't been accessed in a long time.
void CachePruning::prune() {
  struct stat StatBuf;
  llvm::SmallString<128> TimestampFile(Path);
  llvm::sys::path::append(TimestampFile, "llvmcache.timestamp");

  // Try to stat() the timestamp file.
  if (::stat(TimestampFile.c_str(), &StatBuf)) {
    // If the timestamp file wasn't there, create one now.
    if (errno == ENOENT) {
      writeTimestampFile(TimestampFile);
    }
    return;
  }

  // Check whether the time stamp is older than our pruning interval.
  // If not, do nothing.
  time_t TimeStampModTime = StatBuf.st_mtime;
  time_t CurrentTime = time(nullptr);
  if (CurrentTime - TimeStampModTime <= time_t(Interval))
    return;

  // Write a new timestamp file so that nobody else attempts to prune.
  // There is a benign race condition here, if two Clang instances happen to
  // notice at the same time that the timestamp is out-of-date.
  writeTimestampFile(TimestampFile);

  bool ShouldComputeSize = false;
  if (PercentageOfFreeSpace > 0 && PercentageOfFreeSpace < 100)
    ShouldComputeSize = true;

  // Keep track of space
  std::set<std::pair<uint64_t, std::string>> FileSizes;
  uint64_t TotalSize = 0;

  // Walk the entire directory cache, looking for unused files.
  std::error_code EC;
  SmallString<128> CachePathNative;
  llvm::sys::path::native(Path, CachePathNative);
  // Walk all of the files within this directory.
  for (llvm::sys::fs::directory_iterator File(CachePathNative, EC), FileEnd;
       File != FileEnd && !EC; File.increment(EC)) {
    // Do not touch the timestamp.
    if (File->path() == TimestampFile)
      continue;

    // Look at this file. If we can't stat it, there's nothing interesting
    // there.
    if (::stat(File->path().c_str(), &StatBuf))
      continue;

    if (ShouldComputeSize) {
      TotalSize += StatBuf.st_size;
      FileSizes.insert(
          std::make_pair(StatBuf.st_size, std::string(File->path())));
    }

    if (Expiration <= 0)
      continue;

    // If the file has been used recently enough, leave it there.
    time_t FileAccessTime = StatBuf.st_atime;
    if (CurrentTime - FileAccessTime <= time_t(Expiration)) {
      continue;
    }

    // Remove the file.
    llvm::sys::fs::remove(File->path());
  }

  if (ShouldComputeSize) {
    struct statfs statf;
    statfs(".", &statf);
    auto FreeSpace = ((uint64_t)statf.f_bfree) * statf.f_bsize;
    auto FileAndSize = FileSizes.rbegin();
    while (((100 * TotalSize) / FreeSpace) > PercentageOfFreeSpace &&
           FileAndSize != FileSizes.rend()) {
      // Remove the file.
      llvm::sys::fs::remove(FileAndSize->second);
      // Update size
      TotalSize -= FileAndSize->first;
      FreeSpace += FileAndSize->first;
      ++FileAndSize;
    }
  }
}
