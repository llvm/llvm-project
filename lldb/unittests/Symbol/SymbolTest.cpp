//===-- SymbolTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Symbol.h"
#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

static void EncodeDecode(const Symbol &object, const SectionList *sect_list,
                         ByteOrder byte_order) {
  const uint8_t addr_size = 8;
  DataEncoder file(byte_order, addr_size);
  DataEncoder strtab_encoder(byte_order, addr_size);
  ConstStringTable const_strtab;
  object.Encode(file, const_strtab);
  llvm::ArrayRef<uint8_t> bytes = file.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);

  const_strtab.Encode(strtab_encoder);
  llvm::ArrayRef<uint8_t> strtab_bytes = strtab_encoder.GetData();
  DataExtractor strtab_data(strtab_bytes.data(), strtab_bytes.size(),
                            byte_order, addr_size);
  StringTableReader strtab_reader;
  offset_t strtab_data_offset = 0;
  ASSERT_EQ(strtab_reader.Decode(strtab_data, &strtab_data_offset), true);

  Symbol decoded_object;
  offset_t data_offset = 0;
  decoded_object.Decode(data, &data_offset, sect_list, strtab_reader);
  EXPECT_EQ(object, decoded_object);
}

static void EncodeDecode(const Symbol &object, const SectionList *sect_list) {
  EncodeDecode(object, sect_list, eByteOrderLittle);
  EncodeDecode(object, sect_list, eByteOrderBig);
}

TEST(SymbolTest, EncodeDecodeSymbol) {

  SectionSP sect_sp(new Section(
      /*module_sp=*/ModuleSP(),
      /*obj_file=*/nullptr,
      /*sect_id=*/1,
      /*name=*/ConstString(".text"),
      /*sect_type=*/eSectionTypeCode,
      /*file_vm_addr=*/0x1000,
      /*vm_size=*/0x1000,
      /*file_offset=*/0,
      /*file_size=*/0,
      /*log2align=*/5,
      /*flags=*/0x10203040));

  SectionList sect_list;
  sect_list.AddSection(sect_sp);

  Symbol symbol(
      /*symID=*/0x10203040,
      /*name=*/"main",
      /*type=*/eSymbolTypeCode,
      /*bool external=*/false,
      /*bool is_debug=*/false,
      /*bool is_trampoline=*/false,
      /*bool is_artificial=*/false,
      /*section_sp=*/sect_sp,
      /*offset=*/0x0,
      /*size=*/0x100,
      /*size_is_valid=*/true,
      /*contains_linker_annotations=*/false,
      /*flags=*/0x11223344);

  // Test encoding a symbol with an address.
  EncodeDecode(symbol, &sect_list);

  // Test that encoding the bits in the bitfield works for all endianness
  // combos.

  // Test Symbol.m_is_synthetic
  symbol.SetIsSynthetic(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetIsSynthetic(false);

  // Test Symbol.m_is_debug
  symbol.SetDebug(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetDebug(false);

  // Test Symbol.m_is_external
  symbol.SetExternal(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetExternal(false);

  // Test Symbol.m_size_is_sibling
  symbol.SetSizeIsSibling(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetSizeIsSibling(false);

  // Test Symbol.m_size_is_synthesized
  symbol.SetSizeIsSynthesized(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetSizeIsSynthesized(false);

  // Test Symbol.m_size_is_synthesized
  symbol.SetByteSize(0);
  EncodeDecode(symbol, &sect_list);
  symbol.SetByteSize(0x100);

  // Test Symbol.m_demangled_is_synthesized
  symbol.SetDemangledNameIsSynthesized(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetDemangledNameIsSynthesized(false);

  // Test Symbol.m_contains_linker_annotations
  symbol.SetContainsLinkerAnnotations(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetContainsLinkerAnnotations(false);

  // Test Symbol.m_is_weak
  symbol.SetIsWeak(true);
  EncodeDecode(symbol, &sect_list);
  symbol.SetIsWeak(false);

  // Test encoding a symbol with no address.
  symbol.GetAddressRef().SetSection(SectionSP());
  EncodeDecode(symbol, &sect_list);
}
