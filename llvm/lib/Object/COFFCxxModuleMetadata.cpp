//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFFCxxModuleMetadata.h"
#include "llvm/Support/ErrorExtras.h"

namespace llvm::object {

COFFCxxModuleMetadataReader::COFFCxxModuleMetadataReader(
    const COFFCxxModuleMetadata &Map)
    : ModuleData(Map.ModuleData), NamesData(Map.NamesData),
      ModuleIndexWidth(Map.ModuleIndexWidth),
      SymbolIndexWidth(Map.SymbolIndexWidth) {}

bool COFFCxxModuleMetadataReader::hasModuleData() const {
  return !ModuleData.empty();
}

Expected<uint32_t> COFFCxxModuleMetadataReader::readModuleID() {
  switch (ModuleIndexWidth) {
  case 1: {
    if (ModuleData.size() < 1)
      return createStringError("Not enough data");
    uint8_t ID = static_cast<uint8_t>(ModuleData[0]);
    ModuleData = ModuleData.slice(1, StringRef::npos);
    if (ID == std::numeric_limits<uint8_t>::max())
      return std::numeric_limits<uint32_t>::max();
    return ID;
  }
  case 2: {
    if (ModuleData.size() < 2)
      return createStringError("Not enough data");
    uint16_t ID =
        support::endian::read<uint16_t>(ModuleData.data(), endianness::little);
    ModuleData = ModuleData.slice(2, StringRef::npos);
    if (ID == std::numeric_limits<uint16_t>::max())
      return std::numeric_limits<uint32_t>::max();
    return ID;
  }
  case 4: {
    if (ModuleData.size() < 4)
      return createStringError("Not enough data");
    uint32_t ID =
        support::endian::read<uint32_t>(ModuleData.data(), endianness::little);
    ModuleData = ModuleData.slice(4, StringRef::npos);
    return ID;
  }
  default:
    return createStringErrorV("Unsupported index width: {0}", ModuleIndexWidth);
  }
}

Expected<StringRef> COFFCxxModuleMetadataReader::readModuleName() {
  size_t End = NamesData.find('\0');
  if (End == StringRef::npos)
    return createStringError("Missing null terminator");
  StringRef Str = NamesData.slice(0, End);
  NamesData = NamesData.drop_front(End + 1);
  return Str;
}

Expected<ArrayRef<uint8_t>>
COFFCxxModuleMetadataReader::readListImpl(uint8_t Width) {
  StringRef Sentinel;
  switch (Width) {
  case 1:
    Sentinel = "\xff";
    break;
  case 2:
    Sentinel = "\xff\xff";
    break;
  case 4:
    Sentinel = "\xff\xff\xff\xff";
    break;
  default:
    return createStringErrorV("Unsupported index width: {0}", Width);
  }
  size_t Last = ModuleData.find(Sentinel);
  if (Last == StringRef::npos)
    return createStringError("Missing end sentinel");

  ArrayRef<uint8_t> Data(ModuleData.bytes_begin(),
                         ModuleData.bytes_begin() + Last);
  ModuleData = ModuleData.drop_front(Last + Width);
  return Data;
}

Expected<COFFCxxModuleMetadata>
parseCOFFCxxModuleMetadata(StringRef SectionData) {
  if (SectionData.size() <= sizeof(COFFCxxModuleMetadataHeader))
    return createStringError("Insufficient size");

  const auto *Header =
      reinterpret_cast<const COFFCxxModuleMetadataHeader *>(SectionData.data());
  if (Header->Version != 1)
    return createStringError("Unsupported version");

  auto IsSupportedIndexWidth = [](uint8_t Width) {
    return Width == 1 || Width == 2 || Width == 4;
  };

  if (!IsSupportedIndexWidth(Header->ModuleIndexWidth))
    return createStringErrorV("Unsupported module index width: {0}",
                              Header->ModuleIndexWidth);
  if (!IsSupportedIndexWidth(Header->SymbolIndexWidth))
    return createStringErrorV("Unsupported symbol index width: {0}",
                              Header->SymbolIndexWidth);

  size_t ModuleDataSize = Header->ModuleDataSize.value();
  if (ModuleDataSize <= sizeof(COFFCxxModuleMetadataHeader) ||
      ModuleDataSize + 1 >= SectionData.size())
    return createStringErrorV("Invalid module data size: {0}", ModuleDataSize);

  COFFCxxModuleMetadata Map;
  Map.Version = Header->Version;
  Map.Reserved = Header->Reserved;
  Map.ModuleIndexWidth = Header->ModuleIndexWidth;
  Map.SymbolIndexWidth = Header->SymbolIndexWidth;
  Map.ModuleData =
      SectionData.slice(sizeof(COFFCxxModuleMetadataHeader), ModuleDataSize);
  // Skip one reserved(?) byte.
  Map.NamesData = SectionData.slice(ModuleDataSize + 1, StringRef::npos);
  return Map;
}

} // namespace llvm::object
