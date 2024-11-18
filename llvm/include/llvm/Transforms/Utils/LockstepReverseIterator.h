//===- LockstepReverseIterator.h ------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOCKSTEPREVERSEITERATOR_H
#define LLVM_TRANSFORMS_UTILS_LOCKSTEPREVERSEITERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

struct NoActiveBlocksOption {
  template <typename... Args> NoActiveBlocksOption(Args...) {}
};

struct ActiveBlocksOption {
protected:
  SmallSetVector<BasicBlock *, 4> ActiveBlocks;

public:
  ActiveBlocksOption() = default;
};

/// Iterates through instructions in a set of blocks in reverse order from the
/// first non-terminator. For example (assume all blocks have size n):
///   LockstepReverseIterator I([B1, B2, B3]);
///   *I-- = [B1[n], B2[n], B3[n]];
///   *I-- = [B1[n-1], B2[n-1], B3[n-1]];
///   *I-- = [B1[n-2], B2[n-2], B3[n-2]];
///   ...
///
/// The iterator continues processing until all blocks have been exhausted if \p
/// EarlyFailure is explicitly set to \c false. Use \c getActiveBlocks() to
/// determine which blocks are still going and the order they appear in the list
/// returned by operator*.
template <bool EarlyFailure = true>
class LockstepReverseIterator
    : public std::conditional_t<EarlyFailure, NoActiveBlocksOption,
                                ActiveBlocksOption> {
private:
  using BasicBlockT = BasicBlock;
  using InstructionT = Instruction;

  ArrayRef<BasicBlockT *> Blocks;
  SmallVector<InstructionT *, 4> Insts;
  bool Fail;

public:
  LockstepReverseIterator(ArrayRef<BasicBlockT *> Blocks) : Blocks(Blocks) {
    reset();
  }

  void reset() {
    Fail = false;
    if constexpr (!EarlyFailure) {
      this->ActiveBlocks.clear();
      for (BasicBlockT *BB : Blocks)
        this->ActiveBlocks.insert(BB);
    }
    Insts.clear();
    for (BasicBlockT *BB : Blocks) {
      InstructionT *Prev = BB->getTerminator()->getPrevNonDebugInstruction();
      if (!Prev) {
        // Block wasn't big enough - only contained a terminator.
        if constexpr (EarlyFailure) {
          Fail = true;
          return;
        } else {
          this->ActiveBlocks.remove(BB);
          continue;
        }
      }
      Insts.push_back(Prev);
    }
    if (Insts.empty())
      Fail = true;
  }

  bool isValid() const { return !Fail; }
  ArrayRef<InstructionT *> operator*() const { return Insts; }

  // Note: This needs to return a SmallSetVector as the elements of
  // ActiveBlocks will be later copied to Blocks using std::copy. The
  // resultant order of elements in Blocks needs to be deterministic.
  // Using SmallPtrSet instead causes non-deterministic order while
  // copying. And we cannot simply sort Blocks as they need to match the
  // corresponding Values.
  template <bool C = EarlyFailure, std::enable_if_t<!C, int> = 0>
  SmallSetVector<BasicBlockT *, 4> &getActiveBlocks() {
    return this->ActiveBlocks;
  }

  template <bool C = EarlyFailure, std::enable_if_t<!C, int> = 0>
  void restrictToBlocks(SmallSetVector<BasicBlockT *, 4> &Blocks) {
    for (auto It = Insts.begin(); It != Insts.end();) {
      if (!Blocks.contains((*It)->getParent())) {
        this->ActiveBlocks.remove((*It)->getParent());
        It = Insts.erase(It);
      } else {
        ++It;
      }
    }
  }

  void operator--() {
    if (Fail)
      return;
    SmallVector<InstructionT *, 4> NewInsts;
    for (InstructionT *Inst : Insts) {
      InstructionT *Prev = Inst->getPrevNonDebugInstruction();
      if (!Prev) {
        if constexpr (!EarlyFailure) {
          this->ActiveBlocks.remove(Inst->getParent());
        } else {
          Fail = true;
          return;
        }
      } else {
        NewInsts.push_back(Prev);
      }
    }
    if (NewInsts.empty()) {
      Fail = true;
      return;
    }
    Insts = NewInsts;
  }

  void operator++() {
    if (Fail)
      return;
    SmallVector<InstructionT *, 4> NewInsts;
    for (InstructionT *Inst : Insts) {
      InstructionT *Next = Inst->getNextNonDebugInstruction();
      // Already at end of block.
      if (!Next) {
        Fail = true;
        return;
      }
      NewInsts.push_back(Next);
    }
    if (NewInsts.empty()) {
      Fail = true;
      return;
    }
    Insts = NewInsts;
  }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOCKSTEPREVERSEITERATOR_H
