//===- PredIteratorCache.h - pred_iterator Cache ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PredIteratorCache class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PREDITERATORCACHE_H
#define LLVM_IR_PREDITERATORCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

/// PredIteratorCache - This class is an extremely trivial cache for
/// predecessor iterator queries.  This is useful for code that repeatedly
/// wants the predecessor list for the same blocks.
class PredIteratorCache {
  /// Cached list of predecessors, allocated in Memory.
  DenseMap<BasicBlock *, ArrayRef<BasicBlock *>> BlockToPredsMap;

  /// Memory - This is the space that holds cached preds.
  BumpPtrAllocator Memory;

public:
  size_t size(BasicBlock *BB) { return get(BB).size(); }
  ArrayRef<BasicBlock *> get(BasicBlock *BB) {
    ArrayRef<BasicBlock *> &Entry = BlockToPredsMap[BB];
    if (Entry.data())
      return Entry;

    SmallVector<BasicBlock *, 32> PredCache(predecessors(BB));
    BasicBlock **Data = Memory.Allocate<BasicBlock *>(PredCache.size());
    std::copy(PredCache.begin(), PredCache.end(), Data);
    Entry = ArrayRef(Data, PredCache.size());
    return Entry;
  }

  /// clear - Remove all information.
  void clear() {
    BlockToPredsMap.clear();
    Memory.Reset();
  }
};

} // end namespace llvm

#endif
