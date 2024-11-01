//===-- StructuralHash.cpp - IR Hash for expensive checks -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifdef EXPENSIVE_CHECKS

#include "llvm/IR/StructuralHash.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace details {

// Basic hashing mechanism to detect structural change to the IR, used to verify
// pass return status consistency with actual change. Loosely copied from
// llvm/lib/Transforms/Utils/FunctionComparator.cpp

class StructuralHash {
  uint64_t Hash = 0x6acaa36bef8325c5ULL;

  void update(uint64_t V) { Hash = hashing::detail::hash_16_bytes(Hash, V); }

public:
  StructuralHash() = default;

  void update(const Function &F) {
    if (F.empty())
      return;

    update(F.isVarArg());
    update(F.arg_size());

    SmallVector<const BasicBlock *, 8> BBs;
    SmallPtrSet<const BasicBlock *, 16> VisitedBBs;

    BBs.push_back(&F.getEntryBlock());
    VisitedBBs.insert(BBs[0]);
    while (!BBs.empty()) {
      const BasicBlock *BB = BBs.pop_back_val();
      update(45798); // Block header
      for (auto &Inst : *BB)
        update(Inst.getOpcode());

      const Instruction *Term = BB->getTerminator();
      for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
        if (!VisitedBBs.insert(Term->getSuccessor(i)).second)
          continue;
        BBs.push_back(Term->getSuccessor(i));
      }
    }
  }

  void update(const Module &M) {
    for (const Function &F : M)
      update(F);
  }

  uint64_t getHash() const { return Hash; }
};

} // namespace details

uint64_t llvm::StructuralHash(const Function &F) {
  ::details::StructuralHash H;
  H.update(F);
  return H.getHash();
}

uint64_t llvm::StructuralHash(const Module &M) {
  ::details::StructuralHash H;
  H.update(M);
  return H.getHash();
}

#endif
