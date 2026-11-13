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
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
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
