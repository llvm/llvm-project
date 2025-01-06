//===- BasicBlock.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_BASICBLOCK_H
#define LLVM_SANDBOXIR_BASICBLOCK_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/SandboxIR/Value.h"

namespace llvm::sandboxir {

class BasicBlock;
class Function;
class Instruction;

/// Iterator for `Instruction`s in a `BasicBlock.
/// \Returns an sandboxir::Instruction & when derereferenced.
class BBIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Instruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  llvm::BasicBlock *BB;
  llvm::BasicBlock::iterator It;
  Context *Ctx;
  pointer getInstr(llvm::BasicBlock::iterator It) const;

public:
  BBIterator() : BB(nullptr), Ctx(nullptr) {}
  BBIterator(llvm::BasicBlock *BB, llvm::BasicBlock::iterator It, Context *Ctx)
      : BB(BB), It(It), Ctx(Ctx) {}
  reference operator*() const { return *getInstr(It); }
  BBIterator &operator++();
  BBIterator operator++(int) {
    auto Copy = *this;
    ++*this;
    return Copy;
  }
  BBIterator &operator--();
  BBIterator operator--(int) {
    auto Copy = *this;
    --*this;
    return Copy;
  }
  bool operator==(const BBIterator &Other) const {
    assert(Ctx == Other.Ctx && "BBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const BBIterator &Other) const { return !(*this == Other); }
  /// \Returns the SBInstruction that corresponds to this iterator, or null if
  /// the instruction is not found in the IR-to-SandboxIR tables.
  pointer get() const { return getInstr(It); }
  /// \Returns the parent BB.
  BasicBlock *getNodeParent() const;
};

/// Contains a list of sandboxir::Instruction's.
class BasicBlock : public Value {
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildBasicBlockFromLLVMIR(llvm::BasicBlock *LLVMBB);
  friend class Context;     // For `buildBasicBlockFromIR`
  friend class Instruction; // For LLVM Val.

  BasicBlock(llvm::BasicBlock *BB, Context &SBCtx)
      : Value(ClassID::Block, BB, SBCtx) {
    buildBasicBlockFromLLVMIR(BB);
  }

public:
  ~BasicBlock() = default;
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == Value::ClassID::Block;
  }
  Function *getParent() const;
  using iterator = BBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<llvm::BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctx);
  }
  std::reverse_iterator<iterator> rbegin() const {
    return std::make_reverse_iterator(end());
  }
  std::reverse_iterator<iterator> rend() const {
    return std::make_reverse_iterator(begin());
  }
  Context &getContext() const { return Ctx; }
  Instruction *getTerminator() const;
  bool empty() const { return begin() == end(); }
  Instruction &front() const;
  Instruction &back() const;

#ifndef NDEBUG
  void verify() const final;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_BASICBLOCK_H
