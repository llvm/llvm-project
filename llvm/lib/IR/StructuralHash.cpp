//===-- StructuralHash.cpp - IR Hashing -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructuralHash.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

// Basic hashing mechanism to detect structural change to the IR, used to verify
// pass return status consistency with actual change. In addition to being used
// by the MergeFunctions pass.

class StructuralHashImpl {
  stable_hash Hash = 4;

  bool DetailedHash;

  // This random value acts as a block header, as otherwise the partition of
  // opcodes into BBs wouldn't affect the hash, only the order of the opcodes.
  static constexpr stable_hash BlockHeaderHash = 45798;
  static constexpr stable_hash FunctionHeaderHash = 0x62642d6b6b2d6b72;
  static constexpr stable_hash GlobalHeaderHash = 23456;

  // This will produce different values on 32-bit and 64-bit systens as
  // hash_combine returns a size_t. However, this is only used for
  // detailed hashing which, in-tree, only needs to distinguish between
  // differences in functions.
  // TODO: This is not stable.
  template <typename T> stable_hash hashArbitaryType(const T &V) {
    return hash_combine(V);
  }

  stable_hash hashType(Type *ValueType) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(ValueType->getTypeID());
    if (ValueType->isIntegerTy())
      Hashes.emplace_back(ValueType->getIntegerBitWidth());
    return stable_hash_combine(Hashes);
  }

public:
  StructuralHashImpl() = delete;
  explicit StructuralHashImpl(bool DetailedHash) : DetailedHash(DetailedHash) {}

  stable_hash hashConstant(Constant *C) {
    SmallVector<stable_hash> Hashes;
    // TODO: hashArbitaryType() is not stable.
    if (ConstantInt *ConstInt = dyn_cast<ConstantInt>(C)) {
      Hashes.emplace_back(hashArbitaryType(ConstInt->getValue()));
    } else if (ConstantFP *ConstFP = dyn_cast<ConstantFP>(C)) {
      Hashes.emplace_back(hashArbitaryType(ConstFP->getValue()));
    } else if (Function *Func = dyn_cast<Function>(C)) {
      // Hashing the name will be deterministic as LLVM's hashing infrastructure
      // has explicit support for hashing strings and will not simply hash
      // the pointer.
      Hashes.emplace_back(hashArbitaryType(Func->getName()));
    }

    return stable_hash_combine(Hashes);
  }

  stable_hash hashValue(Value *V) {
    // Check constant and return its hash.
    Constant *C = dyn_cast<Constant>(V);
    if (C)
      return hashConstant(C);

    // Hash argument number.
    SmallVector<stable_hash> Hashes;
    if (Argument *Arg = dyn_cast<Argument>(V))
      Hashes.emplace_back(Arg->getArgNo());

    return stable_hash_combine(Hashes);
  }

  stable_hash hashOperand(Value *Operand) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(hashType(Operand->getType()));
    Hashes.emplace_back(hashValue(Operand));
    return stable_hash_combine(Hashes);
  }

  stable_hash hashInstruction(const Instruction &Inst) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Inst.getOpcode());

    if (!DetailedHash)
      return stable_hash_combine(Hashes);

    Hashes.emplace_back(hashType(Inst.getType()));

    // Handle additional properties of specific instructions that cause
    // semantic differences in the IR.
    if (const auto *ComparisonInstruction = dyn_cast<CmpInst>(&Inst))
      Hashes.emplace_back(ComparisonInstruction->getPredicate());

    for (const auto &Op : Inst.operands())
      Hashes.emplace_back(hashOperand(Op));

    return stable_hash_combine(Hashes);
  }

  // A function hash is calculated by considering only the number of arguments
  // and whether a function is varargs, the order of basic blocks (given by the
  // successors of each basic block in depth first order), and the order of
  // opcodes of each instruction within each of these basic blocks. This mirrors
  // the strategy FunctionComparator::compare() uses to compare functions by
  // walking the BBs in depth first order and comparing each instruction in
  // sequence. Because this hash currently does not look at the operands, it is
  // insensitive to things such as the target of calls and the constants used in
  // the function, which makes it useful when possibly merging functions which
  // are the same modulo constants and call targets.
  //
  // Note that different users of StructuralHash will want different behavior
  // out of it (i.e., MergeFunctions will want something different from PM
  // expensive checks for pass modification status). When modifying this
  // function, most changes should be gated behind an option and enabled
  // selectively.
  void update(const Function &F) {
    // Declarations don't affect analyses.
    if (F.isDeclaration())
      return;

    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Hash);
    Hashes.emplace_back(FunctionHeaderHash);

    Hashes.emplace_back(F.isVarArg());
    Hashes.emplace_back(F.arg_size());

    SmallVector<const BasicBlock *, 8> BBs;
    SmallPtrSet<const BasicBlock *, 16> VisitedBBs;

    // Walk the blocks in the same order as
    // FunctionComparator::cmpBasicBlocks(), accumulating the hash of the
    // function "structure." (BB and opcode sequence)
    BBs.push_back(&F.getEntryBlock());
    VisitedBBs.insert(BBs[0]);
    while (!BBs.empty()) {
      const BasicBlock *BB = BBs.pop_back_val();

      Hashes.emplace_back(BlockHeaderHash);
      for (auto &Inst : *BB)
        Hashes.emplace_back(hashInstruction(Inst));

      for (const BasicBlock *Succ : successors(BB))
        if (VisitedBBs.insert(Succ).second)
          BBs.push_back(Succ);
    }

    // Update the combined hash in place.
    Hash = stable_hash_combine(Hashes);
  }

  void update(const GlobalVariable &GV) {
    // Declarations and used/compiler.used don't affect analyses.
    // Since there are several `llvm.*` metadata, like `llvm.embedded.object`,
    // we ignore anything with the `.llvm` prefix
    if (GV.isDeclaration() || GV.getName().starts_with("llvm."))
      return;
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Hash);
    Hashes.emplace_back(GlobalHeaderHash);
    Hashes.emplace_back(GV.getValueType()->getTypeID());

    // Update the combined hash in place.
    Hash = stable_hash_combine(Hashes);
  }

  void update(const Module &M) {
    for (const GlobalVariable &GV : M.globals())
      update(GV);
    for (const Function &F : M)
      update(F);
  }

  uint64_t getHash() const { return Hash; }
};

} // namespace

stable_hash llvm::StructuralHash(const Function &F, bool DetailedHash) {
  StructuralHashImpl H(DetailedHash);
  H.update(F);
  return H.getHash();
}

stable_hash llvm::StructuralHash(const Module &M, bool DetailedHash) {
  StructuralHashImpl H(DetailedHash);
  H.update(M);
  return H.getHash();
}
