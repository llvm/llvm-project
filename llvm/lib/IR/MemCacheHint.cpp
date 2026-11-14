//===- MemCacheHint.cpp - Cache hint metadata helpers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MemCacheHint.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include <optional>
#include <utility>

using namespace llvm;

MDNode *llvm::getMemCacheHint(const Instruction &I, unsigned OperandNo) {
  MDNode *MD = I.getMetadata(LLVMContext::MD_mem_cache_hint);
  if (!MD)
    return nullptr;

  for (unsigned Idx = 0; Idx + 1 < MD->getNumOperands(); Idx += 2)
    if (mdconst::extract<ConstantInt>(MD->getOperand(Idx))->getZExtValue() ==
        OperandNo)
      return cast<MDNode>(MD->getOperand(Idx + 1));
  return nullptr;
}

void llvm::setMemCacheHint(Instruction &I, unsigned OperandNo, MDNode *Hint) {
  SmallVector<std::pair<unsigned, MDNode *>, 2> Entries;
  if (MDNode *MD = I.getMetadata(LLVMContext::MD_mem_cache_hint))
    for (unsigned Idx = 0; Idx + 1 < MD->getNumOperands(); Idx += 2)
      Entries.emplace_back(
          mdconst::extract<ConstantInt>(MD->getOperand(Idx))->getZExtValue(),
          cast<MDNode>(MD->getOperand(Idx + 1)));

  auto It = llvm::lower_bound(Entries, OperandNo,
                              [](const auto &Entry, unsigned OperandNo) {
                                return Entry.first < OperandNo;
                              });
  if (It != Entries.end() && It->first == OperandNo) {
    if (Hint)
      It->second = Hint;
    else
      Entries.erase(It);
  } else if (Hint) {
    Entries.insert(It, {OperandNo, Hint});
  } else {
    return;
  }

  SmallVector<Metadata *, 4> Operands;
  for (const auto &[Index, Payload] : Entries) {
    Operands.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(I.getContext()), Index)));
    Operands.push_back(Payload);
  }
  I.setMetadata(LLVMContext::MD_mem_cache_hint,
                Operands.empty() ? nullptr
                                 : MDNode::get(I.getContext(), Operands));
}

// The supported accesses have one memory operand. Its position also
// distinguishes reads (0) from writes (1).
static std::optional<unsigned> getPointerOperandIndex(const Instruction &I) {
  if (isa<LoadInst>(I))
    return LoadInst::getPointerOperandIndex();
  if (isa<StoreInst>(I))
    return StoreInst::getPointerOperandIndex();
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return std::nullopt;
  switch (II->getIntrinsicID()) {
  case Intrinsic::masked_load:
  case Intrinsic::masked_gather:
    return 0;
  case Intrinsic::masked_store:
  case Intrinsic::masked_scatter:
    return 1;
  default:
    return std::nullopt;
  }
}

static MDNode *getCommonCacheHint(const Instruction &Dest,
                                  ArrayRef<Value *> Sources) {
  std::optional<unsigned> PointerIndex = getPointerOperandIndex(Dest);
  if (!PointerIndex || Sources.empty())
    return nullptr;

  const auto *First = dyn_cast<Instruction>(Sources.front());
  if (!First)
    return nullptr;
  MDNode *MD = First->getMetadata(LLVMContext::MD_mem_cache_hint);
  if (!MD)
    return nullptr;

  for (Value *V : Sources) {
    const auto *Source = dyn_cast<Instruction>(V);
    if (!Source || getPointerOperandIndex(*Source) != PointerIndex ||
        Source->getMetadata(LLVMContext::MD_mem_cache_hint) != MD)
      return nullptr;
  }
  return MD;
}

void llvm::propagateMemCacheHint(Instruction &Dest, ArrayRef<Value *> Sources) {
  Dest.setMetadata(LLVMContext::MD_mem_cache_hint,
                   getCommonCacheHint(Dest, Sources));
}
