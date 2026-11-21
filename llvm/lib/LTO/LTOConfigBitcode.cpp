//===- LTOConfigBitcode.cpp - lto::Config in bitcode ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes and decodes the lto::Config JSON blob in ThinLTO summary indexes.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"

#include "JSONConfigUtils.h"

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;

void encodePipelineTuningOptions(JSONWriter &Writer,
                                 const PipelineTuningOptions &PTO) {
  [[maybe_unused]] const auto &[
#define LLVM_PTO_FIELD(Type, Name, Initializer) , Name
#define LLVM_PTO_FIRST_FIELD(Type, Name, Initializer) Name
#include "llvm/Passes/PipelineTuningOptionsFields.def"
#undef LLVM_PTO_FIRST_FIELD
#undef LLVM_PTO_FIELD
  ] = PTO;

#define LLVM_PTO_FIELD(Type, Name, Initializer)                                \
  encodeField(Writer, "pto." #Name, PTO.Name);
#define LLVM_PTO_FIRST_FIELD LLVM_PTO_FIELD
#include "llvm/Passes/PipelineTuningOptionsFields.def"
#undef LLVM_PTO_FIRST_FIELD
#undef LLVM_PTO_FIELD
}

template <typename T>
void encodeConfigField(JSONWriter &Writer, StringRef Key, const T &Field) {
  encodeField(Writer, Key, Field);
}

void encodeConfigField(JSONWriter &Writer, StringRef Key,
                       const TargetOptions &Field) {
  Writer.putValue(Key, encodeTargetOptionsJSON(Field));
}

void encodeConfigField(JSONWriter &Writer, StringRef Key,
                       std::optional<Reloc::Model> Field) {
  Writer.putValue(Key, Field ? json::Value(static_cast<int32_t>(*Field))
                             : json::Value(nullptr));
}

void encodeConfigField(JSONWriter &Writer, StringRef Key,
                       std::optional<uint64_t> Field) {
  Writer.putValue(Key, json::toJSON(Field));
}

void encodeConfigFields(JSONWriter &Writer, const Config &C) {
  // Catch fields added outside ConfigFields.def unless explicitly accounted
  // for as process-local fields.
  [[maybe_unused]] const auto &[
#define LLVM_LTO_CONFIG_FIELD(Type, Name, Initializer) Name,
#define LLVM_LTO_CONFIG_PTO_FIELD(Type, Name) Name,
#include "llvm/LTO/ConfigFields.def"
#undef LLVM_LTO_CONFIG_PTO_FIELD
#undef LLVM_LTO_CONFIG_FIELD
      LibraryOptions, LoadedPassPlugins, PassBuilderCallback, DiagHandler,
      ResolutionFile, PreOptModuleHook, PostPromoteModuleHook,
      PostInternalizeModuleHook, PostImportModuleHook, PostOptModuleHook,
      PreCodeGenModuleHook, CombinedIndexHook, PostRegularLTOLinkHook,
      GetSummaryIndexOutputStream, GetImportsListOutputArray,
      GetCacheKeyOutputString] = C;

#define LLVM_LTO_CONFIG_FIELD(Type, Name, Initializer)                         \
  encodeConfigField(Writer, #Name, C.Name);
#define LLVM_LTO_CONFIG_PTO_FIELD(Type, Name)                                  \
  encodePipelineTuningOptions(Writer, C.Name);
#include "llvm/LTO/ConfigFields.def"
#undef LLVM_LTO_CONFIG_PTO_FIELD
#undef LLVM_LTO_CONFIG_FIELD
}

template <typename T>
Error decodeConfigField(T &Field, JSONEntryApplier Applier) {
  return decodeField(Field, Applier);
}

Error decodeConfigField(std::optional<Reloc::Model> &Field,
                        JSONEntryApplier Applier) {
  if (Applier.Entry.getAsNull()) {
    Field.reset();
    return Error::success();
  }
  return Applier.applyI32(
      [&](int32_t V) { Field = static_cast<Reloc::Model>(V); });
}

Error decodeConfigField(TargetOptions &Field, JSONEntryApplier Applier) {
  auto Options = decodeTargetOptionsJSON(Applier.Entry);
  if (!Options)
    return Options.takeError();
  Field = std::move(*Options);
  return Error::success();
}

Error decodeConfigField(std::optional<uint64_t> &Field,
                        JSONEntryApplier Applier) {
  const json::Value &Entry = Applier.Entry;
  if (Entry.getAsNull()) {
    Field.reset();
    return Error::success();
  }
  auto Value = Entry.getAsUINT64();
  if (!Value)
    return configError(
        "RemarksHotnessThreshold: expected an unsigned integer or null");
  Field = *Value;
  return Error::success();
}

Error applyPipelineTuningOption(PipelineTuningOptions &PTO, StringRef Key,
                                JSONEntryApplier Applier) {
#define LLVM_PTO_FIELD(Type, Name, Initializer)                                \
  if (Key == "pto." #Name)                                                     \
    return decodeField(PTO.Name, Applier);
#define LLVM_PTO_FIRST_FIELD LLVM_PTO_FIELD
#include "llvm/Passes/PipelineTuningOptionsFields.def"
#undef LLVM_PTO_FIRST_FIELD
#undef LLVM_PTO_FIELD

  return configError("unknown lto config key: " + Key);
}

Error applyEntry(Config &C, StringRef Key, JSONEntryApplier Applier) {
#define LLVM_LTO_CONFIG_FIELD(Type, Name, Initializer)                         \
  if (Key == #Name)                                                            \
    return decodeConfigField(C.Name, Applier);
#define LLVM_LTO_CONFIG_PTO_FIELD(Type, Name)                                  \
  if (Key.starts_with("pto."))                                                 \
    return applyPipelineTuningOption(C.Name, Key, Applier);
#include "llvm/LTO/ConfigFields.def"
#undef LLVM_LTO_CONFIG_PTO_FIELD
#undef LLVM_LTO_CONFIG_FIELD

  return configError("unknown lto config key: " + Key);
}

Expected<Config> decodeConfigJSON(StringRef Text) {
  auto Value = json::parse(Text);
  if (!Value)
    return Value.takeError();
  return decodeVersionedJSON<Config>(
      *Value, kVersion, "lto config",
      [](Config &C, StringRef Key, const json::Value &Entry) {
        return applyEntry(C, Key, JSONEntryApplier{Entry, Key});
      });
}

} // namespace

std::string lto::serializeLTOConfig(const Config &Config) {
  json::Object Object{{"Version", kVersion}};
  JSONWriter Writer(Object);
  encodeConfigFields(Writer, Config);

  std::string Text;
  raw_string_ostream OS(Text);
  OS << json::Value(std::move(Object));
  return Text;
}

Expected<std::optional<Config>>
lto::readLTOConfigFromSummaryIndexIfPresent(const ModuleSummaryIndex &Index) {
  if (StringRef JSON = Index.getLTOConfigJSON(); !JSON.empty())
    return decodeConfigJSON(JSON);
  return std::nullopt;
}
