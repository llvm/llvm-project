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
  if (ModuleIndexWidth != 1 && ModuleIndexWidth != 2 && ModuleIndexWidth != 4)
    return createStringError("unsupported index width: %d", ModuleIndexWidth);

  if (ModuleData.size() < ModuleIndexWidth)
    return createStringError("not enough data");

  uint32_t ID = std::numeric_limits<uint32_t>::max();
  switch (ModuleIndexWidth) {
  case 1: {
    uint8_t V = static_cast<uint8_t>(ModuleData[0]);
    if (V != std::numeric_limits<uint8_t>::max())
      ID = V;
  } break;
  case 2: {
    uint16_t V =
        support::endian::read<uint16_t>(ModuleData.data(), endianness::little);
    if (V != std::numeric_limits<uint16_t>::max())
      ID = V;
  } break;
  case 4: {
    ID = support::endian::read<uint32_t>(ModuleData.data(), endianness::little);
  } break;
  }

  ModuleData = ModuleData.slice(ModuleIndexWidth, StringRef::npos);
  return ID;
}

Expected<StringRef> COFFCxxModuleMetadataReader::readModuleName() {
  size_t End = NamesData.find('\0');
  if (End == StringRef::npos)
    return createStringError("missing null terminator");
  StringRef Str = NamesData.slice(0, End);
  NamesData = NamesData.drop_front(End + 1);
  return Str;
}

Expected<ArrayRef<uint8_t>>
COFFCxxModuleMetadataReader::readListImpl(uint8_t Width) {
  if (Width != 1 && Width != 2 && Width != 4)
    return createStringError("unsupported index width: %d", Width);

  StringRef Sentinel("\xff\xff\xff\xff", Width);
  StringRef Rest = ModuleData;
  while (Rest.size() >= Width) {
    if (Rest.consume_front(Sentinel)) {
      ArrayRef<uint8_t> Data(ModuleData.bytes_begin(),
                             Rest.bytes_begin() - Width);
      ModuleData = Rest;
      return Data;
    }
    Rest = Rest.drop_front(Width);
  }

  return createStringError("missing end sentinel");
}

Expected<COFFCxxModuleMetadata>
parseCOFFCxxModuleMetadata(StringRef SectionData) {
  if (SectionData.size() <= sizeof(COFFCxxModuleMetadataHeader))
    return createStringError("insufficient size: got %d, expected more than %d",
                             SectionData.size(),
                             sizeof(COFFCxxModuleMetadataHeader));

  const auto *Header =
      reinterpret_cast<const COFFCxxModuleMetadataHeader *>(SectionData.data());
  if (Header->Version != 1)
    return createStringError("unsupported version: %d", Header->Version);

  auto IsSupportedIndexWidth = [](uint8_t Width) {
    return Width == 1 || Width == 2 || Width == 4;
  };

  if (!IsSupportedIndexWidth(Header->ModuleIndexWidth))
    return createStringErrorV("unsupported module index width: {0}",
                              Header->ModuleIndexWidth);
  if (!IsSupportedIndexWidth(Header->SymbolIndexWidth))
    return createStringErrorV("unsupported symbol index width: {0}",
                              Header->SymbolIndexWidth);

  size_t ModuleDataSize = Header->ModuleDataSize.value();
  if (ModuleDataSize <= sizeof(COFFCxxModuleMetadataHeader))
    return createStringError(
        "module data size too small: got %d, expected more than %d",
        ModuleDataSize, sizeof(COFFCxxModuleMetadataHeader));
  if (ModuleDataSize + 1 >= SectionData.size())
    return createStringErrorV(
        "module data size too big: got %d, section size is %d", ModuleDataSize,
        SectionData.size());

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
