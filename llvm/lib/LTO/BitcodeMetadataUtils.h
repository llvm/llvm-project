//===- BitcodeMetadataUtils.h - shared LTO metadata helpers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal helpers shared by LTOConfigBitcode.cpp and TargetOptionsBitcode.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_LTO_BITCODEMETADATAUTILS_H
#define LLVM_LIB_LTO_BITCODEMETADATAUTILS_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace llvm {
namespace lto {
namespace bitcodemeta {

inline Error metadataError(const Twine &Msg) {
  return make_error<StringError>(Msg.str(), inconvertibleErrorCode());
}

inline Metadata *getI32Value(LLVMContext &Ctx, int32_t V) {
  return ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt32Ty(Ctx), V));
}

inline Metadata *getI64Value(LLVMContext &Ctx, uint64_t V) {
  return ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Ctx), V));
}

inline Metadata *getStringValue(LLVMContext &Ctx, StringRef S) {
  return MDString::get(Ctx, S);
}

class MetadataWriter {
  SmallVectorImpl<Metadata *> &Out;
  LLVMContext &Ctx;

public:
  MetadataWriter(SmallVectorImpl<Metadata *> &Out, LLVMContext &Ctx)
      : Out(Out), Ctx(Ctx) {}

  LLVMContext &getContext() const { return Ctx; }

  void putEntry(StringRef Key, Metadata *Value) {
    Metadata *Ops[] = {getStringValue(Ctx, Key), Value};
    Out.push_back(MDNode::get(Ctx, Ops));
  }

  void putI32(StringRef Key, int32_t V) { putEntry(Key, getI32Value(Ctx, V)); }

  void putI64(StringRef Key, uint64_t V) { putEntry(Key, getI64Value(Ctx, V)); }

  void putBool(StringRef Key, bool V) { putI32(Key, V ? 1 : 0); }

  void putString(StringRef Key, StringRef V) {
    if (!V.empty())
      putEntry(Key, getStringValue(Ctx, V));
  }

  void putNode(StringRef Key, MDNode *Node) { putEntry(Key, Node); }

  void putStringList(StringRef Key, ArrayRef<std::string> Values) {
    if (Values.empty())
      return;
    SmallVector<Metadata *, 8> Elems;
    for (const std::string &S : Values)
      Elems.push_back(getStringValue(Ctx, S));
    putEntry(Key, MDNode::get(Ctx, Elems));
  }
};

inline Expected<int32_t> getI32Field(const MDNode &Entry,
                                     StringRef EntryKind = "metadata entry") {
  if (Entry.getNumOperands() != 2)
    return metadataError(EntryKind + " must have 2 operands");
  auto *Val = mdconst::dyn_extract<ConstantInt>(Entry.getOperand(1));
  if (!Val || !Val->getType()->isIntegerTy(32))
    return metadataError(EntryKind + " value must be i32");
  return static_cast<int32_t>(Val->getSExtValue());
}

inline Expected<int64_t> getI64Field(const MDNode &Entry,
                                     StringRef EntryKind = "metadata entry") {
  if (Entry.getNumOperands() != 2)
    return metadataError(EntryKind + " must have 2 operands");
  auto *Val = mdconst::dyn_extract<ConstantInt>(Entry.getOperand(1));
  if (!Val || !Val->getType()->isIntegerTy(64))
    return metadataError(EntryKind + " value must be i64");
  return static_cast<int64_t>(Val->getSExtValue());
}

inline Expected<StringRef>
getStringField(const MDNode &Entry, StringRef EntryKind = "metadata entry") {
  if (Entry.getNumOperands() != 2)
    return metadataError(EntryKind + " must have 2 operands");
  auto *Val = dyn_cast<MDString>(Entry.getOperand(1));
  if (!Val)
    return metadataError(EntryKind + " value must be a string");
  return Val->getString();
}

inline Expected<std::vector<std::string>>
getStringListField(const MDNode &Entry,
                   StringRef EntryKind = "metadata entry") {
  if (Entry.getNumOperands() != 2)
    return metadataError(EntryKind + " must have 2 operands");
  auto *List = dyn_cast<MDNode>(Entry.getOperand(1));
  if (!List)
    return metadataError(EntryKind + " value must be a string list");
  std::vector<std::string> Out;
  Out.reserve(List->getNumOperands());
  for (Metadata *Op : List->operands()) {
    auto *S = dyn_cast<MDString>(Op);
    if (!S)
      return metadataError(EntryKind + " string list element must be a string");
    Out.push_back(S->getString().str());
  }
  return Out;
}

inline Expected<MDNode *> getNodeField(const MDNode &Entry,
                                       StringRef EntryKind = "metadata entry") {
  if (Entry.getNumOperands() != 2)
    return metadataError(EntryKind + " must have 2 operands");
  auto *Node = dyn_cast<MDNode>(Entry.getOperand(1));
  if (!Node)
    return metadataError(EntryKind + " value must be a metadata node");
  return Node;
}

struct EntryApplier {
  const MDNode &Entry;
  StringRef EntryKind;

  Error applyI32(function_ref<void(int32_t)> Setter) {
    auto V = getI32Field(Entry, EntryKind);
    if (!V)
      return V.takeError();
    Setter(*V);
    return Error::success();
  }

  Error applyI64(function_ref<void(int64_t)> Setter) {
    auto V = getI64Field(Entry, EntryKind);
    if (!V)
      return V.takeError();
    Setter(*V);
    return Error::success();
  }

  Error applyBool(function_ref<void(bool)> Setter) {
    auto V = getI32Field(Entry, EntryKind);
    if (!V)
      return V.takeError();
    if (*V != 0 && *V != 1)
      return metadataError(EntryKind + " boolean value must be 0 or 1");
    Setter(*V != 0);
    return Error::success();
  }

  Error applyString(function_ref<void(StringRef)> Setter) {
    auto V = getStringField(Entry, EntryKind);
    if (!V)
      return V.takeError();
    Setter(*V);
    return Error::success();
  }

  Error applyStringList(function_ref<void(std::vector<std::string>)> Setter) {
    auto V = getStringListField(Entry, EntryKind);
    if (!V)
      return V.takeError();
    Setter(std::move(*V));
    return Error::success();
  }
};

template <typename T, typename ApplyEntryFn>
Expected<T>
decodeVersionedMetadata(const MDNode *Root, unsigned ExpectedVersion,
                        StringRef RootKind, ApplyEntryFn ApplyEntry) {
  if (!Root || Root->getNumOperands() < 1)
    return metadataError("malformed " + RootKind + " metadata root");

  auto *VersionVal = mdconst::dyn_extract<ConstantInt>(Root->getOperand(0));
  if (!VersionVal || !VersionVal->getType()->isIntegerTy(32))
    return metadataError("malformed " + RootKind + " metadata version");
  if (VersionVal->getZExtValue() != ExpectedVersion)
    return metadataError("unsupported " + RootKind + " metadata version");

  T Result{};
  for (unsigned I = 1; I < Root->getNumOperands(); ++I) {
    auto *Entry = dyn_cast<MDNode>(Root->getOperand(I));
    if (!Entry || Entry->getNumOperands() != 2)
      return metadataError("malformed " + RootKind + " metadata entry");
    auto *KeyMD = dyn_cast<MDString>(Entry->getOperand(0));
    if (!KeyMD)
      return metadataError(RootKind + " key must be a string");
    if (Error E = ApplyEntry(Result, KeyMD->getString(), *Entry))
      return std::move(E);
  }
  return Result;
}

} // namespace bitcodemeta
} // namespace lto
} // namespace llvm

#endif
