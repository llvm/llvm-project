//===-- PISACacheCtrlMMRA.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISACacheCtrlMMRA.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
constexpr StringLiteral CacheCtrlMMRAPrefix = "pisa.cache.ctrl";
} // namespace

std::optional<unsigned> PISA::getCacheCtrlFromMMRA(const Instruction &I) {
  MMRAMetadata MMRA(I);
  SmallVector<unsigned, 2> Values;
  for (const MMRAMetadata::TagT &Tag : MMRA) {
    StringRef Prefix = Tag.first;
    StringRef Suffix = Tag.second;
    if (Prefix != CacheCtrlMMRAPrefix)
      continue;
    unsigned Value;
    if (Suffix.getAsInteger(10, Value))
      return std::nullopt;
    if (!is_contained(Values, Value))
      Values.push_back(Value);
  }
  if (Values.empty())
    return std::nullopt;
  if (Values.size() > 1) {
    SmallString<128> Msg;
    raw_svector_ostream OS(Msg);
    OS << "instruction has conflicting pisa.cache.ctrl MMRA tags: {";
    interleaveComma(Values, OS);
    OS << "}; keeping " << Values.front();
    I.getContext().diagnose(
        DiagnosticInfoGeneric(&I, Twine(StringRef(Msg)), DS_Warning));
  }
  return Values.front();
}

void PISA::setCacheCtrlMMRA(Instruction &I, unsigned Value) {
  LLVMContext &Ctx = I.getContext();

  SmallVector<MMRAMetadata::TagT, 4> Tags;
  MMRAMetadata Existing(I);
  for (const MMRAMetadata::TagT &Tag : Existing) {
    if (Tag.first != CacheCtrlMMRAPrefix)
      Tags.push_back(Tag);
  }

  SmallString<8> Buf;
  Tags.emplace_back(CacheCtrlMMRAPrefix, Twine(Value).toStringRef(Buf));

  I.setMetadata(LLVMContext::MD_mmra, MMRAMetadata::getMD(Ctx, Tags));
}

void PISA::copyCacheCtrlMMRA(const Instruction &From, Instruction &To) {
  std::optional<unsigned> Value = getCacheCtrlFromMMRA(From);
  if (Value)
    setCacheCtrlMMRA(To, *Value);
}
