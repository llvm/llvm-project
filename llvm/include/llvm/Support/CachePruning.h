//=- CachePruning.h - Helper to manage the pruning of a cache dir -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pruning of a directory intended for cache storage, using
// various policies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CACHE_PRUNING_H
#define LLVM_SUPPORT_CACHE_PRUNING_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

/// Handle pruning a directory provided a path and some options to control what
/// to prune.
class CachePruning {
public:
  /// Prepare to prune \p Path.
  CachePruning(StringRef Path) : Path(Path) {}

  /// Define the pruning interval. This is intended to be used to avoid scanning
  /// the directory too often. It does not impact the decision of which file to
  /// prune. A value of 0 forces the scan to occurs.
  CachePruning &setPruningInterval(int PruningInterval) {
    Interval = PruningInterval;
    return *this;
  }

  /// Define the expiration for a file. When a file hasn't been accessed for
  /// \p ExpireAfter seconds, it is removed from the cache. A value of 0 disable
  /// the expiration-based pruning.
  CachePruning &setEntryExpiration(unsigned ExpireAfter) {
    Expiration = ExpireAfter;
    return *this;
  }

  /// Define the maximum size for the cache directory, in terms of percentage of
  /// the available space on the the disk. Set to 100 to indicate no limit, 50
  /// to indicate that the cache size will not be left over half the
  /// available disk space. A value over 100 will be reduced to 100. A value of
  /// 0 disable the size-based pruning.
  CachePruning &setMaxSize(unsigned Percentage) {
    PercentageOfAvailableSpace = std::min(100u, Percentage);
    return *this;
  }

  /// Peform pruning using the supplied options, returns true if pruning
  /// occured, i.e. if PruningInterval was expired.
  bool prune();

private:
  // Options that matches the setters above.
  std::string Path;
  unsigned Expiration = 0;
  unsigned Interval = 0;
  unsigned PercentageOfAvailableSpace = 0;
};

} // namespace llvm

#endif