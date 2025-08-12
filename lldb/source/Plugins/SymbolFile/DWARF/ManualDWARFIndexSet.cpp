//===-- ManualDWARFIndex.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/ManualDWARFIndexSet.h"
#include "lldb/Core/DataFileCache.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include <cstdint>

using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

namespace {
// Define IDs for the different tables when encoding and decoding the
// ManualDWARFIndex NameToDIE objects so we can avoid saving any empty maps.
enum DataID {
  kDataIDFunctionBasenames = 1u,
  kDataIDFunctionFullnames,
  kDataIDFunctionMethods,
  kDataIDFunctionSelectors,
  kDataIDFunctionObjcClassSelectors,
  kDataIDGlobals,
  kDataIDTypes,
  kDataIDNamespaces,
  kDataIDEnd = 255u,
};
} // namespace

// Version 2 changes the encoding of DIERef objects used in the DWARF manual
// index name tables. See DIERef class for details.
static constexpr uint32_t CURRENT_CACHE_VERSION = 2;

static constexpr llvm::StringLiteral kIdentifierManualDWARFIndex("DIDX");

std::optional<IndexSet<NameToDIE>>
plugin::dwarf::DecodeIndexSet(const DataExtractor &data,
                              lldb::offset_t *offset_ptr) {
  StringTableReader strtab;
  // We now decode the string table for all strings in the data cache file.
  if (!strtab.Decode(data, offset_ptr))
    return std::nullopt;

  llvm::StringRef identifier((const char *)data.GetData(offset_ptr, 4), 4);
  if (identifier != kIdentifierManualDWARFIndex)
    return std::nullopt;
  const uint32_t version = data.GetU32(offset_ptr);
  if (version != CURRENT_CACHE_VERSION)
    return std::nullopt;

  IndexSet<NameToDIE> result;
  while (true) {
    switch (data.GetU8(offset_ptr)) {
    default:
      // If we got here, this is not expected, we expect the data IDs to match
      // one of the values from the DataID enumeration.
      return std::nullopt;
    case kDataIDFunctionBasenames:
      if (!result.function_basenames.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDFunctionFullnames:
      if (!result.function_fullnames.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDFunctionMethods:
      if (!result.function_methods.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDFunctionSelectors:
      if (!result.function_selectors.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDFunctionObjcClassSelectors:
      if (!result.objc_class_selectors.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDGlobals:
      if (!result.globals.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDTypes:
      if (!result.types.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDNamespaces:
      if (!result.namespaces.Decode(data, offset_ptr, strtab))
        return std::nullopt;
      break;
    case kDataIDEnd:
      // We got to the end of our NameToDIE encodings.
      return std::move(result);
      break;
    }
  }
}

void plugin::dwarf::EncodeIndexSet(const IndexSet<NameToDIE> &set,
                                   DataEncoder &encoder) {
  ConstStringTable strtab;

  // Encoder the DWARF index into a separate encoder first. This allows us
  // gather all of the strings we willl need in "strtab" as we will need to
  // write the string table out before the symbol table.
  DataEncoder index_encoder(encoder.GetByteOrder(),
                            encoder.GetAddressByteSize());

  index_encoder.AppendData(kIdentifierManualDWARFIndex);
  // Encode the data version.
  index_encoder.AppendU32(CURRENT_CACHE_VERSION);

  if (!set.function_basenames.IsEmpty()) {
    index_encoder.AppendU8(kDataIDFunctionBasenames);
    set.function_basenames.Encode(index_encoder, strtab);
  }
  if (!set.function_fullnames.IsEmpty()) {
    index_encoder.AppendU8(kDataIDFunctionFullnames);
    set.function_fullnames.Encode(index_encoder, strtab);
  }
  if (!set.function_methods.IsEmpty()) {
    index_encoder.AppendU8(kDataIDFunctionMethods);
    set.function_methods.Encode(index_encoder, strtab);
  }
  if (!set.function_selectors.IsEmpty()) {
    index_encoder.AppendU8(kDataIDFunctionSelectors);
    set.function_selectors.Encode(index_encoder, strtab);
  }
  if (!set.objc_class_selectors.IsEmpty()) {
    index_encoder.AppendU8(kDataIDFunctionObjcClassSelectors);
    set.objc_class_selectors.Encode(index_encoder, strtab);
  }
  if (!set.globals.IsEmpty()) {
    index_encoder.AppendU8(kDataIDGlobals);
    set.globals.Encode(index_encoder, strtab);
  }
  if (!set.types.IsEmpty()) {
    index_encoder.AppendU8(kDataIDTypes);
    set.types.Encode(index_encoder, strtab);
  }
  if (!set.namespaces.IsEmpty()) {
    index_encoder.AppendU8(kDataIDNamespaces);
    set.namespaces.Encode(index_encoder, strtab);
  }
  index_encoder.AppendU8(kDataIDEnd);

  // Now that all strings have been gathered, we will emit the string table.
  strtab.Encode(encoder);
  // Followed by the symbol table data.
  encoder.AppendData(index_encoder.GetData());
}
