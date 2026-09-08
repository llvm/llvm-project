//===- TargetOptionsBitcode.cpp - TargetOptions in bitcode ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes llvm::TargetOptions as module metadata that is stored in bitcode.
//
// Layout:
//   !llvm.lto.target_options = !{ !0 }
//   !0 = !{ i32 <version>, !1, !2, ... }
//   !1 = !{ !"<key>", <value> }
//
// Value kinds:
//   - i32 ConstantInt for bools, enums, and small integers
//   - MDString for std::string fields
//   - nested MDNode for structured fields such as MemoryBuffer
//
// Fields that cannot be represented in IR (such as callbacks) are
// intentionally omitted.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/TargetOptionsBitcode.h"

#include "BitcodeMetadataUtils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;
constexpr StringRef kEntryKind = "target options entry";

void encodeMemoryBuffer(MetadataWriter &Writer, StringRef Key,
                        const std::shared_ptr<MemoryBuffer> &Buffer) {
  if (!Buffer)
    return;

  Metadata *Fields[] = {
      getStringValue(Writer.getContext(), Buffer->getBufferIdentifier()),
      getStringValue(Writer.getContext(), Buffer->getBuffer())};
  Writer.putNode(Key, MDNode::get(Writer.getContext(), Fields));
}

Error decodeMemoryBuffer(std::shared_ptr<MemoryBuffer> &Buffer,
                         const MDNode &Entry) {
  Expected<MDNode *> Fields = getNodeField(Entry, kEntryKind);
  if (!Fields)
    return Fields.takeError();
  if ((*Fields)->getNumOperands() != 2)
    return metadataError(kEntryKind +
                         " memory buffer must contain an identifier and data");

  auto *Identifier = dyn_cast<MDString>((*Fields)->getOperand(0));
  auto *Data = dyn_cast<MDString>((*Fields)->getOperand(1));
  if (!Identifier || !Data)
    return metadataError(kEntryKind + " memory buffer fields must be strings");

  Buffer = MemoryBuffer::getMemBufferCopy(Data->getString(),
                                          Identifier->getString());
  return Error::success();
}

void encodeMCTargetOptions(MetadataWriter &Writer, const MCTargetOptions &MC) {
#define MC_TARGET_OPTION_ENCODE_BITFIELD(Type, Name, Bits, Default)            \
  Writer.putBool("mc." #Name, MC.Name);
#define MC_TARGET_OPTION_ENCODE_BOOL(Type, Name, Bits, Default)                \
  Writer.putBool("mc." #Name, MC.Name);
#define MC_TARGET_OPTION_ENCODE_ENUM(Type, Name, Bits, Default)                \
  Writer.putI32("mc." #Name, static_cast<int32_t>(MC.Name));
#define MC_TARGET_OPTION_ENCODE_OPTIONAL_UINT(Type, Name, Bits, Default)       \
  if (MC.Name)                                                                 \
    Writer.putU32("mc." #Name, *MC.Name);
#define MC_TARGET_OPTION_ENCODE_INT(Type, Name, Bits, Default)                 \
  Writer.putI32("mc." #Name, MC.Name);
#define MC_TARGET_OPTION_ENCODE_PAIR(Type, Name, Bits, Default)                \
  Writer.putI32("mc." #Name "Major", MC.Name.first);                           \
  Writer.putI32("mc." #Name "Minor", MC.Name.second);
#define MC_TARGET_OPTION_ENCODE_STRING(Type, Name, Bits, Default)              \
  Writer.putString("mc." #Name, MC.Name);
#define MC_TARGET_OPTION_ENCODE_STRING_LIST(Type, Name, Bits, Default)         \
  Writer.putStringList("mc." #Name, MC.Name);
#define MC_TARGET_OPTION(Type, Name, Bits, Default, Kind)                      \
  MC_TARGET_OPTION_ENCODE_##Kind(Type, Name, Bits, Default)
#include "llvm/MC/MCTargetOptions.def"
#undef MC_TARGET_OPTION_ENCODE_BITFIELD
#undef MC_TARGET_OPTION_ENCODE_BOOL
#undef MC_TARGET_OPTION_ENCODE_ENUM
#undef MC_TARGET_OPTION_ENCODE_OPTIONAL_UINT
#undef MC_TARGET_OPTION_ENCODE_INT
#undef MC_TARGET_OPTION_ENCODE_PAIR
#undef MC_TARGET_OPTION_ENCODE_STRING
#undef MC_TARGET_OPTION_ENCODE_STRING_LIST
}

void encodeTargetOptionsFields(MetadataWriter &Writer,
                               const TargetOptions &Opt) {
#define TARGET_OPTION_ENCODE_BOOL(Type, Name, Bits, Default)                   \
  Writer.putBool(#Name, Opt.Name);
#define TARGET_OPTION_ENCODE_U32_BITFIELD(Type, Name, Bits, Default)           \
  Writer.putU32(#Name, Opt.Name);
#define TARGET_OPTION_ENCODE_U32(Type, Name, Bits, Default)                    \
  Writer.putU32(#Name, Opt.Name);
#define TARGET_OPTION_ENCODE_ENUM(Type, Name, Bits, Default)                   \
  Writer.putI32(#Name, static_cast<int32_t>(Opt.Name));
#define TARGET_OPTION_ENCODE_STRING(Type, Name, Bits, Default)                 \
  Writer.putString(#Name, Opt.Name);
#define TARGET_OPTION_ENCODE_PAIR(Type, Name, Bits, Default)                   \
  Writer.putI32(#Name "Major", Opt.Name.first);                                \
  Writer.putI32(#Name "Minor", Opt.Name.second);
#define TARGET_OPTION_ENCODE_BUFFER(Type, Name, Bits, Default)                 \
  encodeMemoryBuffer(Writer, #Name, Opt.Name);
#define TARGET_OPTION_ENCODE_MC(Type, Name, Bits, Default)                     \
  encodeMCTargetOptions(Writer, Opt.Name);
#define TARGET_OPTION(Type, Name, Bits, Default, Kind)                         \
  TARGET_OPTION_ENCODE_##Kind(Type, Name, Bits, Default)
#include "llvm/Target/TargetOptions.def"
#undef TARGET_OPTION_ENCODE_BOOL
#undef TARGET_OPTION_ENCODE_U32_BITFIELD
#undef TARGET_OPTION_ENCODE_U32
#undef TARGET_OPTION_ENCODE_ENUM
#undef TARGET_OPTION_ENCODE_STRING
#undef TARGET_OPTION_ENCODE_PAIR
#undef TARGET_OPTION_ENCODE_BUFFER
#undef TARGET_OPTION_ENCODE_MC
}

Error applyEntry(TargetOptions &Opt, StringRef Key, const MDNode &Entry) {
  EntryApplier Applier{Entry, kEntryKind};

#define TARGET_OPTION_TYPE(...) __VA_ARGS__
#define TARGET_OPTION_DECODE_BOOL(Type, Name, Bits, Default)                   \
  if (Key == #Name)                                                            \
    return Applier.applyBool([&](bool V) { Opt.Name = V; });
#define TARGET_OPTION_DECODE_U32_BITFIELD(Type, Name, Bits, Default)           \
  if (Key == #Name)                                                            \
    return Applier.applyU32([&](uint32_t V) { Opt.Name = V; });
#define TARGET_OPTION_DECODE_U32(Type, Name, Bits, Default)                    \
  if (Key == #Name)                                                            \
    return Applier.applyU32([&](uint32_t V) { Opt.Name = V; });
#define TARGET_OPTION_DECODE_ENUM(Type, Name, Bits, Default)                   \
  if (Key == #Name)                                                            \
    return Applier.applyI32([&](int32_t V) {                                   \
      Opt.Name = static_cast<TARGET_OPTION_TYPE Type>(V);                      \
    });
#define TARGET_OPTION_DECODE_STRING(Type, Name, Bits, Default)                 \
  if (Key == #Name)                                                            \
    return Applier.applyString([&](StringRef V) { Opt.Name = V.str(); });
#define TARGET_OPTION_DECODE_PAIR(Type, Name, Bits, Default)                   \
  if (Key == #Name "Major")                                                    \
    return Applier.applyI32([&](int32_t V) { Opt.Name.first = V; });           \
  if (Key == #Name "Minor")                                                    \
    return Applier.applyI32([&](int32_t V) { Opt.Name.second = V; });
#define TARGET_OPTION_DECODE_BUFFER(Type, Name, Bits, Default)                 \
  if (Key == #Name)                                                            \
    return decodeMemoryBuffer(Opt.Name, Entry);
#define TARGET_OPTION_DECODE_MC(Type, Name, Bits, Default)                     \
  MCTargetOptions &MC = Opt.Name;
#define TARGET_OPTION(Type, Name, Bits, Default, Kind)                         \
  TARGET_OPTION_DECODE_##Kind(Type, Name, Bits, Default)
#include "llvm/Target/TargetOptions.def"
#undef TARGET_OPTION_TYPE
#undef TARGET_OPTION_DECODE_BOOL
#undef TARGET_OPTION_DECODE_U32_BITFIELD
#undef TARGET_OPTION_DECODE_U32
#undef TARGET_OPTION_DECODE_ENUM
#undef TARGET_OPTION_DECODE_STRING
#undef TARGET_OPTION_DECODE_PAIR
#undef TARGET_OPTION_DECODE_BUFFER
#undef TARGET_OPTION_DECODE_MC

#define MC_TARGET_OPTION_TYPE(...) __VA_ARGS__
#define MC_TARGET_OPTION_DECODE_BITFIELD(Type, Name, Bits, Default)            \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyBool([&](bool V) { MC.Name = V; });
#define MC_TARGET_OPTION_DECODE_BOOL(Type, Name, Bits, Default)                \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyBool([&](bool V) { MC.Name = V; });
#define MC_TARGET_OPTION_DECODE_ENUM(Type, Name, Bits, Default)                \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyI32([&](int32_t V) {                                   \
      MC.Name = static_cast<MC_TARGET_OPTION_TYPE Type>(V);                    \
    });
#define MC_TARGET_OPTION_DECODE_OPTIONAL_UINT(Type, Name, Bits, Default)       \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyU32([&](uint32_t V) { MC.Name = V; });
#define MC_TARGET_OPTION_DECODE_INT(Type, Name, Bits, Default)                 \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyI32([&](int32_t V) { MC.Name = V; });
#define MC_TARGET_OPTION_DECODE_PAIR(Type, Name, Bits, Default)                \
  if (Key == "mc." #Name "Major")                                              \
    return Applier.applyI32([&](int32_t V) { MC.Name.first = V; });            \
  if (Key == "mc." #Name "Minor")                                              \
    return Applier.applyI32([&](int32_t V) { MC.Name.second = V; });
#define MC_TARGET_OPTION_DECODE_STRING(Type, Name, Bits, Default)              \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyString([&](StringRef V) { MC.Name = V.str(); });
#define MC_TARGET_OPTION_DECODE_STRING_LIST(Type, Name, Bits, Default)         \
  if (Key == "mc." #Name)                                                      \
    return Applier.applyStringList(                                            \
        [&](std::vector<std::string> V) { MC.Name = std::move(V); });
#define MC_TARGET_OPTION(Type, Name, Bits, Default, Kind)                      \
  MC_TARGET_OPTION_DECODE_##Kind(Type, Name, Bits, Default)
#include "llvm/MC/MCTargetOptions.def"
#undef MC_TARGET_OPTION_TYPE
#undef MC_TARGET_OPTION_DECODE_BITFIELD
#undef MC_TARGET_OPTION_DECODE_BOOL
#undef MC_TARGET_OPTION_DECODE_ENUM
#undef MC_TARGET_OPTION_DECODE_OPTIONAL_UINT
#undef MC_TARGET_OPTION_DECODE_INT
#undef MC_TARGET_OPTION_DECODE_PAIR
#undef MC_TARGET_OPTION_DECODE_STRING
#undef MC_TARGET_OPTION_DECODE_STRING_LIST

  return metadataError("unknown target options key: " + Key);
}

} // namespace

bool lto::hasEncodedTargetOptions(const Module &M) {
  return M.getNamedMetadata(TargetOptionsMetadataName) != nullptr;
}

Error lto::encodeTargetOptionsToModule(Module &M,
                                       const TargetOptions &Options) {
  MDNode *Root = encodeTargetOptionsAsNode(M.getContext(), Options);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(TargetOptionsMetadataName);
  NMD->clearOperands();
  NMD->addOperand(Root);
  return Error::success();
}

MDNode *lto::encodeTargetOptionsAsNode(LLVMContext &Ctx,
                                       const TargetOptions &Options) {
  SmallVector<Metadata *, 32> Entries;
  Entries.push_back(getI32Value(Ctx, kVersion));
  MetadataWriter Writer(Entries, Ctx);
  encodeTargetOptionsFields(Writer, Options);
  return MDNode::get(Ctx, Entries);
}

Expected<TargetOptions> lto::decodeTargetOptionsFromNode(const MDNode *Root) {
  return decodeVersionedMetadata<TargetOptions>(
      Root, kVersion, "target options",
      [](TargetOptions &Opt, StringRef Key, const MDNode &Entry) {
        return applyEntry(Opt, Key, Entry);
      });
}

Expected<TargetOptions> lto::decodeTargetOptionsFromModule(const Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata(TargetOptionsMetadataName);
  if (!NMD || NMD->getNumOperands() == 0)
    return metadataError("missing target options metadata");

  return decodeTargetOptionsFromNode(dyn_cast<MDNode>(NMD->getOperand(0)));
}
