//===-- PISACacheCtrlMMRA.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H
#define LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H

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

#include <optional>

namespace llvm {
namespace PISA {

inline constexpr StringRef CacheCtrlMMRAPrefix = "pisa.cache.ctrl";

// Reads the cache-ctrl integer from the "pisa.cache.ctrl" MMRA tag on I.
// Returns std::nullopt if the tag is absent or malformed. If more than
// one distinct cache-ctrl value is attached (e.g. a passthrough merge of
// two memory ops with different cache hints by an upstream pass such as
// the LoadStoreVectorizer), emits a warning through the LLVMContext
// listing the conflicting values and keeps one of them so the compiler
// can still produce valid output.
inline std::optional<unsigned> getCacheCtrlFromMMRA(const Instruction &I) {
  MMRAMetadata MMRA(I);
  SmallVector<unsigned, 2> Values;
  for (const auto &[Prefix, Suffix] : MMRA) {
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

// Sets the "pisa.cache.ctrl" MMRA tag on I to Value. Preserves all
// other MMRA tags already on I, replacing any prior cache-ctrl tag.
inline void setCacheCtrlMMRA(Instruction &I, unsigned Value) {
  LLVMContext &Ctx = I.getContext();

  SmallVector<MMRAMetadata::TagT, 4> Tags;
  MMRAMetadata Existing(I);
  for (const auto &Tag : Existing) {
    if (Tag.first != CacheCtrlMMRAPrefix)
      Tags.push_back(Tag);
  }

  SmallString<8> Buf;
  Tags.emplace_back(CacheCtrlMMRAPrefix, Twine(Value).toStringRef(Buf));

  I.setMetadata(LLVMContext::MD_mmra, MMRAMetadata::getMD(Ctx, Tags));
}

// Copies the "pisa.cache.ctrl" MMRA tag from From to To if present,
// preserving all other MMRA tags already on To.
inline void copyCacheCtrlMMRA(const Instruction &From, Instruction &To) {
  if (auto Value = getCacheCtrlFromMMRA(From))
    setCacheCtrlMMRA(To, *Value);
}

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISACACHECTRLMMRA_H
