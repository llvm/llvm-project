//===- LTOConfigBitcode.cpp - lto::Config in bitcode ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encodes serializable lto::Config fields as module metadata in bitcode.
//
// Layout:
//   !llvm.lto.config = !{ !0 }
//   !0 = !{ i32 <version>, !1, !2, ... }
//   !1 = !{ !"<key>", <value> }
//
// Value kinds:
//   - i32 / i64 ConstantInt for scalars
//   - MDString for strings
//   - MDNode list of MDStrings for vector<string>
//   - nested MDNode for TargetOptions (via encodeTargetOptionsAsNode)
//
// Omitted fields (process-local / non-data):
//   LoadedPassPlugins, PreCodeGenPassesHook, DiagHandler, ResolutionFile,
//   PreOptModuleHook, PostPromoteModuleHook, PostInternalizeModuleHook,
//   PostImportModuleHook, PostOptModuleHook, PreCodeGenModuleHook,
//   CombinedIndexHook, GetSummaryIndexOutputStream, GetImportsListOutputArray,
//   GetCacheKeyOutputString
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"

#include "BitcodeMetadataUtils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/LTO/TargetOptionsBitcode.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::lto;
using namespace llvm::lto::bitcodemeta;

namespace {

constexpr unsigned kVersion = 1;
constexpr StringRef kEntryKind = "lto config entry";

Error writeConfigBitcode(raw_ostream &Out, const Config &Config) {
  LLVMContext Ctx;
  Module M("llvm.lto.config", Ctx);
  if (Error Err = encodeLTOConfigToModule(M, Config))
    return Err;
  WriteBitcodeToFile(M, Out);
  return Error::success();
}

Expected<std::optional<Config>>
readConfigBitcodeIfPresent(MemoryBufferRef Buffer) {
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> M = parseBitcodeFile(Buffer, Ctx);
  if (!M)
    return M.takeError();
  if (!hasEncodedLTOConfig(**M))
    return std::nullopt;
  return decodeLTOConfigFromModule(**M);
}

Expected<Config> readConfigBitcode(MemoryBufferRef Buffer) {
  Expected<std::optional<Config>> Config = readConfigBitcodeIfPresent(Buffer);
  if (!Config)
    return Config.takeError();
  if (!*Config)
    return metadataError("missing lto config metadata");
  return std::move(**Config);
}

void encodePipelineTuningOptions(MetadataWriter &Writer,
                                 const PipelineTuningOptions &PTO) {
#define PIPELINE_TUNING_OPTION_BOOL(Name)                                      \
  Writer.putBool("pto." #Name, PTO.Name);
#define PIPELINE_TUNING_OPTION_I32(Name) Writer.putI32("pto." #Name, PTO.Name);
#define PIPELINE_TUNING_OPTION_U32(Name) Writer.putU32("pto." #Name, PTO.Name);
#define PIPELINE_TUNING_OPTION(Type, Name, Default, Kind)                      \
  PIPELINE_TUNING_OPTION_##Kind(Name)
#include "llvm/Passes/PipelineTuningOptions.def"
#undef PIPELINE_TUNING_OPTION_BOOL
#undef PIPELINE_TUNING_OPTION_I32
#undef PIPELINE_TUNING_OPTION_U32
}

void encodeRemarksHotnessThreshold(MetadataWriter &Writer,
                                   const std::optional<uint64_t> &Threshold) {
  int32_t Mode = 0;
  uint64_t Value = 0;
  if (!Threshold.has_value()) {
    Mode = 2; // auto
  } else if (*Threshold == 0) {
    Mode = 0; // disabled
  } else {
    Mode = 1; // manual
    Value = *Threshold;
  }
  LLVMContext &Ctx = Writer.getContext();
  Metadata *Ops[] = {getI32Value(Ctx, Mode), getI64Value(Ctx, Value)};
  Writer.putEntry("RemarksHotnessThreshold", MDNode::get(Ctx, Ops));
}

Error decodeRemarksHotnessThreshold(std::optional<uint64_t> &Threshold,
                                    const MDNode &Entry) {
  auto Node = getNodeField(Entry, kEntryKind);
  if (!Node)
    return Node.takeError();
  if ((*Node)->getNumOperands() != 2)
    return metadataError("RemarksHotnessThreshold must have mode and value");
  auto *Mode = mdconst::dyn_extract<ConstantInt>((*Node)->getOperand(0));
  auto *Value = mdconst::dyn_extract<ConstantInt>((*Node)->getOperand(1));
  if (!Mode || !Mode->getType()->isIntegerTy(32) || !Value ||
      !Value->getType()->isIntegerTy(64))
    return metadataError("malformed RemarksHotnessThreshold metadata");
  switch (Mode->getZExtValue()) {
  case 0:
    Threshold = 0;
    break;
  case 1:
    Threshold = Value->getZExtValue();
    break;
  case 2:
    Threshold = std::nullopt;
    break;
  default:
    return metadataError("invalid RemarksHotnessThreshold mode");
  }
  return Error::success();
}

void encodeConfigFields(MetadataWriter &Writer, const Config &C) {
#define LTO_CONFIG_ENCODE_STRING(Type, Name) Writer.putString(#Name, C.Name);
#define LTO_CONFIG_ENCODE_TARGET_OPTIONS(Type, Name)                           \
  Writer.putNode(#Name, encodeTargetOptionsAsNode(Writer.getContext(), C.Name));
#define LTO_CONFIG_ENCODE_STRING_LIST(Type, Name)                              \
  Writer.putStringList(#Name, C.Name);
#define LTO_CONFIG_ENCODE_NONE(Type, Name)
#define LTO_CONFIG_ENCODE_OPTIONAL_RELOC_MODEL(Type, Name)                     \
  Writer.putBool(#Name ".HasValue", C.Name.has_value());                       \
  if (C.Name)                                                                  \
    Writer.putI32(#Name, static_cast<int32_t>(*C.Name));
#define LTO_CONFIG_ENCODE_OPTIONAL_ENUM(Type, Name)                            \
  if (C.Name)                                                                  \
    Writer.putI32(#Name, static_cast<int32_t>(*C.Name));
#define LTO_CONFIG_ENCODE_ENUM(Type, Name)                                     \
  Writer.putI32(#Name, static_cast<int32_t>(C.Name));
#define LTO_CONFIG_ENCODE_I32(Type, Name)                                      \
  Writer.putI32(#Name, static_cast<int32_t>(C.Name));
#define LTO_CONFIG_ENCODE_U32(Type, Name) Writer.putU32(#Name, C.Name);
#define LTO_CONFIG_ENCODE_BOOL(Type, Name) Writer.putBool(#Name, C.Name);
#define LTO_CONFIG_ENCODE_REMARKS_HOTNESS(Type, Name)                          \
  encodeRemarksHotnessThreshold(Writer, C.Name);
#define LTO_CONFIG_ENCODE_PIPELINE_TUNING_OPTIONS(Type, Name)                  \
  encodePipelineTuningOptions(Writer, C.Name);
#define LTO_CONFIG_OPTION(Type, Name, Default, Kind)                           \
  LTO_CONFIG_ENCODE_##Kind(Type, Name)
#define LTO_CONFIG_MUTABLE_OPTION(Type, Name, Default, Kind)                   \
  LTO_CONFIG_ENCODE_##Kind(Type, Name)
#include "llvm/LTO/Config.def"
#undef LTO_CONFIG_ENCODE_STRING
#undef LTO_CONFIG_ENCODE_TARGET_OPTIONS
#undef LTO_CONFIG_ENCODE_STRING_LIST
#undef LTO_CONFIG_ENCODE_NONE
#undef LTO_CONFIG_ENCODE_OPTIONAL_RELOC_MODEL
#undef LTO_CONFIG_ENCODE_OPTIONAL_ENUM
#undef LTO_CONFIG_ENCODE_ENUM
#undef LTO_CONFIG_ENCODE_I32
#undef LTO_CONFIG_ENCODE_U32
#undef LTO_CONFIG_ENCODE_BOOL
#undef LTO_CONFIG_ENCODE_REMARKS_HOTNESS
#undef LTO_CONFIG_ENCODE_PIPELINE_TUNING_OPTIONS
}

Error applyEntry(Config &C, StringRef Key, const MDNode &Entry) {
  EntryApplier Applier{Entry, kEntryKind};

#define LTO_CONFIG_DECODE_STRING(Type, Name)                                   \
  if (Key == #Name)                                                            \
    return Applier.applyString([&](StringRef V) { C.Name = V.str(); });
#define LTO_CONFIG_DECODE_TARGET_OPTIONS(Type, Name)                           \
  if (Key == #Name) {                                                          \
    auto Node = getNodeField(Entry, kEntryKind);                               \
    if (!Node)                                                                 \
      return Node.takeError();                                                 \
    auto Opt = decodeTargetOptionsFromNode(*Node);                             \
    if (!Opt)                                                                  \
      return Opt.takeError();                                                  \
    C.Name = std::move(*Opt);                                                  \
    return Error::success();                                                   \
  }
#define LTO_CONFIG_DECODE_STRING_LIST(Type, Name)                              \
  if (Key == #Name)                                                            \
    return Applier.applyStringList(                                            \
        [&](std::vector<std::string> V) { C.Name = std::move(V); });
#define LTO_CONFIG_DECODE_NONE(Type, Name)
#define LTO_CONFIG_DECODE_OPTIONAL_RELOC_MODEL(Type, Name)                     \
  if (Key == #Name)                                                            \
    return Applier.applyI32(                                                   \
        [&](int32_t V) { C.Name = static_cast<Reloc::Model>(V); });            \
  if (Key == #Name ".HasValue")                                                \
    return Applier.applyBool([&](bool V) {                                     \
      if (!V)                                                                  \
        C.Name = std::nullopt;                                                 \
    });
#define LTO_CONFIG_DECODE_OPTIONAL_ENUM(Type, Name)                            \
  if (Key == #Name)                                                            \
    return Applier.applyI32([&](int32_t V) {                                   \
      C.Name = static_cast<typename Type::value_type>(V);                      \
    });
#define LTO_CONFIG_DECODE_ENUM(Type, Name)                                     \
  if (Key == #Name)                                                            \
    return Applier.applyI32([&](int32_t V) { C.Name = static_cast<Type>(V); });
#define LTO_CONFIG_DECODE_I32(Type, Name)                                      \
  if (Key == #Name)                                                            \
    return Applier.applyI32([&](int32_t V) { C.Name = static_cast<Type>(V); });
#define LTO_CONFIG_DECODE_U32(Type, Name)                                      \
  if (Key == #Name)                                                            \
    return Applier.applyU32([&](uint32_t V) { C.Name = V; });
#define LTO_CONFIG_DECODE_BOOL(Type, Name)                                     \
  if (Key == #Name)                                                            \
    return Applier.applyBool([&](bool V) { C.Name = V; });
#define LTO_CONFIG_DECODE_REMARKS_HOTNESS(Type, Name)                          \
  if (Key == #Name)                                                            \
    return decodeRemarksHotnessThreshold(C.Name, Entry);
#define LTO_CONFIG_DECODE_PIPELINE_TUNING_OPTIONS(Type, Name)                  \
  PipelineTuningOptions &PTO = C.Name;
#define LTO_CONFIG_OPTION(Type, Name, Default, Kind)                           \
  LTO_CONFIG_DECODE_##Kind(Type, Name)
#define LTO_CONFIG_MUTABLE_OPTION(Type, Name, Default, Kind)                   \
  LTO_CONFIG_DECODE_##Kind(Type, Name)
#include "llvm/LTO/Config.def"
#undef LTO_CONFIG_DECODE_STRING
#undef LTO_CONFIG_DECODE_TARGET_OPTIONS
#undef LTO_CONFIG_DECODE_STRING_LIST
#undef LTO_CONFIG_DECODE_NONE
#undef LTO_CONFIG_DECODE_OPTIONAL_RELOC_MODEL
#undef LTO_CONFIG_DECODE_OPTIONAL_ENUM
#undef LTO_CONFIG_DECODE_ENUM
#undef LTO_CONFIG_DECODE_I32
#undef LTO_CONFIG_DECODE_U32
#undef LTO_CONFIG_DECODE_BOOL
#undef LTO_CONFIG_DECODE_REMARKS_HOTNESS
#undef LTO_CONFIG_DECODE_PIPELINE_TUNING_OPTIONS

#define PIPELINE_TUNING_DECODE_BOOL(Type, Name)                                \
  if (Key == "pto." #Name)                                                     \
    return Applier.applyBool([&](bool V) { PTO.Name = V; });
#define PIPELINE_TUNING_DECODE_I32(Type, Name)                                 \
  if (Key == "pto." #Name)                                                     \
    return Applier.applyI32(                                                   \
        [&](int32_t V) { PTO.Name = static_cast<Type>(V); });
#define PIPELINE_TUNING_DECODE_U32(Type, Name)                                 \
  if (Key == "pto." #Name)                                                     \
    return Applier.applyU32([&](uint32_t V) { PTO.Name = V; });
#define PIPELINE_TUNING_OPTION(Type, Name, Default, Kind)                      \
  PIPELINE_TUNING_DECODE_##Kind(Type, Name)
#include "llvm/Passes/PipelineTuningOptions.def"
#undef PIPELINE_TUNING_DECODE_BOOL
#undef PIPELINE_TUNING_DECODE_I32
#undef PIPELINE_TUNING_DECODE_U32

  return metadataError("unknown lto config key: " + Key);
}

Expected<Config> decodeConfigFromRoot(const MDNode *Root) {
  return decodeVersionedMetadata<Config>(
      Root, kVersion, "lto config",
      [](Config &C, StringRef Key, const MDNode &Entry) {
        return applyEntry(C, Key, Entry);
      });
}

} // namespace

bool lto::hasEncodedLTOConfig(const Module &M) {
  return M.getNamedMetadata(LTOConfigMetadataName) != nullptr;
}

Error lto::encodeLTOConfigToModule(Module &M, const Config &Config) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<Metadata *, 64> Entries;
  Entries.push_back(getI32Value(Ctx, kVersion));
  MetadataWriter Writer(Entries, Ctx);
  encodeConfigFields(Writer, Config);

  MDNode *Root = MDNode::get(Ctx, Entries);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(LTOConfigMetadataName);
  NMD->clearOperands();
  NMD->addOperand(Root);
  return Error::success();
}

Expected<Config> lto::decodeLTOConfigFromModule(const Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata(LTOConfigMetadataName);
  if (!NMD || NMD->getNumOperands() == 0)
    return metadataError("missing lto config metadata");
  return decodeConfigFromRoot(dyn_cast<MDNode>(NMD->getOperand(0)));
}

Error lto::writeLTOConfigToFile(StringRef Path, const Config &Config) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, "cannot open LTO config file '%s'",
                             Path.str().c_str());
  if (Error Err = writeConfigBitcode(OS, Config))
    return Err;
  OS.close();
  if (OS.has_error())
    return createStringError(OS.error(), "cannot write LTO config file '%s'",
                             Path.str().c_str());
  return Error::success();
}

Expected<Config> lto::readLTOConfigFromFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(),
                             "cannot read LTO config file '%s'",
                             Path.str().c_str());

  return readConfigBitcode((*Buffer)->getMemBufferRef());
}

Error lto::writeIndexWithLTOConfigToFile(
    const ModuleSummaryIndex &Index, const Config &Config, raw_ostream &Out,
    const ModuleToSummariesForIndexTy *ModuleToSummariesForIndex,
    const GVSummaryPtrSet *DecSummaries) {
  LLVMContext Ctx;
  Module MetadataModule("llvm.lto.config", Ctx);
  if (Error Err = encodeLTOConfigToModule(MetadataModule, Config))
    return Err;
  writeIndexToFile(Index, Out, ModuleToSummariesForIndex, DecSummaries,
                   &MetadataModule);
  return Error::success();
}

Expected<Config> lto::readLTOConfigFromSummaryIndex(MemoryBufferRef Buffer) {
  return readConfigBitcode(Buffer);
}

Expected<std::optional<Config>>
lto::readLTOConfigFromSummaryIndexIfPresent(MemoryBufferRef Buffer) {
  return readConfigBitcodeIfPresent(Buffer);
}
