//===- TargetOptionsBitcode.cpp - TargetOptions in bitcode ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes and decodes TargetOptions for the LTO configuration JSON.
//
//===----------------------------------------------------------------------===//

#include "JSONConfigUtils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;

using bitcodemeta::decodeField;
using bitcodemeta::encodeField;

void encodeField(JSONWriter &Writer, StringRef Key,
                 const std::shared_ptr<MemoryBuffer> &Buffer) {
  if (Buffer)
    Writer.putValue(Key,
                    json::Object{{"Identifier",
                                  encodeString(Buffer->getBufferIdentifier())},
                                 {"Data", encodeString(Buffer->getBuffer())}});
}

Error decodeField(std::shared_ptr<MemoryBuffer> &Buffer,
                  JSONEntryApplier Applier) {
  const json::Value &Entry = Applier.Entry;
  auto *Object = Entry.getAsObject();
  const auto *Identifier = Object ? Object->get("Identifier") : nullptr;
  const auto *Data = Object ? Object->get("Data") : nullptr;
  if (!Identifier || !Data)
    return configError("expected memory buffer identifier and data");
  auto Name = decodeString(*Identifier);
  if (!Name)
    return Name.takeError();
  auto Bytes = decodeString(*Data);
  if (!Bytes)
    return Bytes.takeError();
  Buffer = MemoryBuffer::getMemBufferCopy(*Bytes, *Name);
  return Error::success();
}

void encodeMCTargetOptions(JSONWriter &Writer, const MCTargetOptions &MC) {
  [[maybe_unused]] const auto &[
#define LLVM_MC_OPTION_FIELD(Type, Name, Initializer) Name,
#define LLVM_MC_OPTION_BITFIELD(Name) Name,
#define LLVM_MC_OPTION_BINUTILS_FIELD(Name) Name,
#include "llvm/MC/MCTargetOptionsFields.def"
#undef LLVM_MC_OPTION_BINUTILS_FIELD
#undef LLVM_MC_OPTION_BITFIELD
#undef LLVM_MC_OPTION_FIELD
      LibraryOptions] = MC;

#define LLVM_MC_OPTION_FIELD(Type, Name, Initializer)                          \
  encodeField(Writer, "mc." #Name, MC.Name);
#define LLVM_MC_OPTION_BITFIELD(Name) LLVM_MC_OPTION_FIELD(bool, Name, )
#define LLVM_MC_OPTION_BINUTILS_FIELD(Name)                                    \
  encodeField(Writer, "mc." #Name "Major", MC.Name.first);                     \
  encodeField(Writer, "mc." #Name "Minor", MC.Name.second);
#include "llvm/MC/MCTargetOptionsFields.def"
#undef LLVM_MC_OPTION_BINUTILS_FIELD
#undef LLVM_MC_OPTION_BITFIELD
#undef LLVM_MC_OPTION_FIELD
}

void encodeTargetOptionsFields(JSONWriter &Writer, const TargetOptions &Opt) {
  [[maybe_unused]] const auto &[
#define LLVM_TARGET_OPTION_FIELD(Type, Name, Initializer) , Name
#define LLVM_TARGET_OPTION_BITFIELD(Name, Width, WireType) , Name
#define LLVM_TARGET_OPTION_FIRST_BITFIELD(Name, Width, WireType) Name
#define LLVM_TARGET_OPTION_MC_FIELD(Name) , Name
#include "llvm/Target/TargetOptionsFields.def"
#undef LLVM_TARGET_OPTION_MC_FIELD
#undef LLVM_TARGET_OPTION_FIRST_BITFIELD
#undef LLVM_TARGET_OPTION_BITFIELD
#undef LLVM_TARGET_OPTION_FIELD
  ] = Opt;

#define LLVM_TARGET_OPTION_FIELD(Type, Name, Initializer)                      \
  encodeField(Writer, #Name, Opt.Name);
#define LLVM_TARGET_OPTION_BITFIELD(Name, Width, WireType)                     \
  encodeField(Writer, #Name, static_cast<WireType>(Opt.Name));
#define LLVM_TARGET_OPTION_FIRST_BITFIELD LLVM_TARGET_OPTION_BITFIELD
#define LLVM_TARGET_OPTION_MC_FIELD(Name)                                      \
  encodeMCTargetOptions(Writer, Opt.Name);
#include "llvm/Target/TargetOptionsFields.def"
#undef LLVM_TARGET_OPTION_MC_FIELD
#undef LLVM_TARGET_OPTION_FIRST_BITFIELD
#undef LLVM_TARGET_OPTION_BITFIELD
#undef LLVM_TARGET_OPTION_FIELD
}

Error applyMCEntry(MCTargetOptions &MC, StringRef Key,
                   JSONEntryApplier Applier) {
#define LLVM_MC_OPTION_FIELD(Type, Name, Initializer)                          \
  if (Key == "mc." #Name)                                                      \
    return decodeField(MC.Name, Applier);
#define LLVM_MC_OPTION_BITFIELD(Name)                                          \
  if (Key == "mc." #Name)                                                      \
    return decodeFieldWith<bool>(Applier, [&](bool V) { MC.Name = V; });
#define LLVM_MC_OPTION_BINUTILS_FIELD(Name)                                    \
  if (Key == "mc." #Name "Major")                                              \
    return decodeField(MC.Name.first, Applier);                                \
  if (Key == "mc." #Name "Minor")                                              \
    return decodeField(MC.Name.second, Applier);
#include "llvm/MC/MCTargetOptionsFields.def"
#undef LLVM_MC_OPTION_BINUTILS_FIELD
#undef LLVM_MC_OPTION_BITFIELD
#undef LLVM_MC_OPTION_FIELD

  return configError("unknown target options key: " + Key);
}

Error applyEntry(TargetOptions &Opt, StringRef Key, JSONEntryApplier Applier) {
#define LLVM_TARGET_OPTION_FIELD(Type, Name, Initializer)                      \
  if (Key == #Name)                                                            \
    return decodeField(Opt.Name, Applier);
#define LLVM_TARGET_OPTION_BITFIELD(Name, Width, WireType)                     \
  if (Key == #Name)                                                            \
    return decodeFieldWith<WireType>(Applier,                                  \
                                     [&](WireType V) { Opt.Name = V; });
#define LLVM_TARGET_OPTION_FIRST_BITFIELD LLVM_TARGET_OPTION_BITFIELD
#define LLVM_TARGET_OPTION_MC_FIELD(Name)                                      \
  if (Key.starts_with("mc."))                                                  \
    return applyMCEntry(Opt.Name, Key, Applier);
#include "llvm/Target/TargetOptionsFields.def"
#undef LLVM_TARGET_OPTION_MC_FIELD
#undef LLVM_TARGET_OPTION_FIRST_BITFIELD
#undef LLVM_TARGET_OPTION_BITFIELD
#undef LLVM_TARGET_OPTION_FIELD

  return configError("unknown target options key: " + Key);
}

} // namespace

json::Object
llvm::lto::bitcodemeta::encodeTargetOptionsJSON(const TargetOptions &Options) {
  json::Object Object{{"Version", kVersion}};
  JSONWriter Writer(Object);
  encodeTargetOptionsFields(Writer, Options);
  return Object;
}

Expected<TargetOptions>
llvm::lto::bitcodemeta::decodeTargetOptionsJSON(const json::Value &Value) {
  return decodeVersionedJSON<TargetOptions>(
      Value, kVersion, "target options",
      [](TargetOptions &Opt, StringRef Key, const json::Value &Entry) {
        return applyEntry(Opt, Key, JSONEntryApplier{Entry, Key});
      });
}
