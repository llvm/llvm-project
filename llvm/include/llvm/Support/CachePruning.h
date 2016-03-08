//===-- Regex.h - Regular Expression matcher implementation -*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pruning of a directory inteded for cache storage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CACHE_PRUNING_H
#define LLVM_SUPPORT_CACHE_PRUNING_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class CachePruning {
public:
  CachePruning(StringRef Path) : Path(Path) {}

  CachePruning &setEntryExpiration(int ExpireAfter) {
    Expiration = ExpireAfter;
    return *this;
  }
  CachePruning &setPruningInterval(int PruningInterval) {
    Interval = PruningInterval;
    return *this;
  }
  CachePruning &setMaxSize(unsigned Percentage) {
    PercentageOfFreeSpace = Percentage;
    return *this;
  }

  void prune();

private:
  std::string Path;
  int Expiration;
  int Interval;
  unsigned PercentageOfFreeSpace;
};

} // namespace llvm

#endif