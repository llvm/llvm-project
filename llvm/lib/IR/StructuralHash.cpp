//===-- StructuralHash.cpp - IR Hashing -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructuralHash.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

// Basic hashing mechanism to detect structural change to the IR, used to verify
// pass return status consistency with actual change. Loosely copied from
// llvm/lib/Transforms/Utils/FunctionComparator.cpp

class StructuralHashImpl {
  hash_code Hash;

  template <typename T> void hash(const T &V) { Hash = hash_combine(Hash, V); }

public:
  StructuralHashImpl() : Hash(4) {}

  void update(const Function &F) {
    // Declarations don't affect analyses.
    if (F.isDeclaration())
      return;

    hash(12345); // Function header

    hash(F.isVarArg());
    hash(F.arg_size());

    SmallVector<const BasicBlock *, 8> BBs;
    SmallPtrSet<const BasicBlock *, 16> VisitedBBs;

    BBs.push_back(&F.getEntryBlock());
    VisitedBBs.insert(BBs[0]);
    while (!BBs.empty()) {
      const BasicBlock *BB = BBs.pop_back_val();
      hash(45798); // Block header
      for (auto &Inst : *BB)
        hash(Inst.getOpcode());

      const Instruction *Term = BB->getTerminator();
      for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
        if (!VisitedBBs.insert(Term->getSuccessor(i)).second)
          continue;
        BBs.push_back(Term->getSuccessor(i));
      }
    }
  }

  void update(const GlobalVariable &GV) {
    // Declarations and used/compiler.used don't affect analyses.
    // Since there are several `llvm.*` metadata, like `llvm.embedded.object`,
    // we ignore anything with the `.llvm` prefix
    if (GV.isDeclaration() || GV.getName().starts_with("llvm."))
      return;
    hash(23456); // Global header
    hash(GV.getValueType()->getTypeID());
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

uint64_t llvm::StructuralHash(const Function &F) {
  StructuralHashImpl H;
  H.update(F);
  return H.getHash();
}

uint64_t llvm::StructuralHash(const Module &M) {
  StructuralHashImpl H;
  H.update(M);
  return H.getHash();
}
