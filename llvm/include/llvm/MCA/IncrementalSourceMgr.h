//===---------------- IncrementalSourceMgr.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains IncrementalSourceMgr, an implementation of SourceMgr
/// that allows users to add new instructions incrementally / dynamically.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_INCREMENTALSOURCEMGR_H
#define LLVM_MCA_INCREMENTALSOURCEMGR_H

#include "llvm/MCA/SourceMgr.h"
#include <deque>

namespace llvm {
namespace mca {

/// An implementation of \a SourceMgr that allows users to add new instructions
/// incrementally / dynamically.
/// Note that this SourceMgr takes ownership of all \a mca::Instruction.
class IncrementalSourceMgr : public SourceMgr {
  /// Owner of all mca::Instruction instances. Note that we use std::deque here
  /// to have a better throughput, in comparison to std::vector or
  /// llvm::SmallVector, as they usually pay a higher re-allocation cost when
  /// there is a large number of instructions.
  std::deque<UniqueInst> InstStorage;

  /// Current instruction index.
  unsigned TotalCounter;

  /// End-of-stream flag.
  bool EOS;

public:
  IncrementalSourceMgr() : TotalCounter(0U), EOS(false) {}

  void clear() {
    InstStorage.clear();
    TotalCounter = 0U;
    EOS = false;
  }

  ArrayRef<UniqueInst> getInstructions() const override {
    llvm_unreachable("Not applicable");
  }

  bool hasNext() const override { return TotalCounter < InstStorage.size(); }
  bool isEnd() const override { return EOS; }

  SourceRef peekNext() const override {
    assert(hasNext());
    return SourceRef(TotalCounter, *InstStorage[TotalCounter]);
  }

  /// Add a new instruction.
  void addInst(UniqueInst &&Inst) { InstStorage.emplace_back(std::move(Inst)); }

  void updateNext() override { ++TotalCounter; }

  /// Mark the end of instruction stream.
  void endOfStream() { EOS = true; }
};

} // end namespace mca
} // end namespace llvm

#endif // LLVM_MCA_INCREMENTALSOURCEMGR_H
