//===-- DWARFIndexCachingTest.cpp -------------------------------------=---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DIERef.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/ManualDWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/NameToDIE.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

static void EncodeDecode(const DIERef &object, ByteOrder byte_order) {
  const uint8_t addr_size = 8;
  DataEncoder encoder(byte_order, addr_size);
  object.Encode(encoder);
  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);
  offset_t data_offset = 0;
  EXPECT_EQ(object, DIERef::Decode(data, &data_offset));
}

static void EncodeDecode(const DIERef &object) {
  EncodeDecode(object, eByteOrderLittle);
  EncodeDecode(object, eByteOrderBig);
}

TEST(DWARFIndexCachingTest, DIERefEncodeDecode) {
  // Tests DIERef::Encode(...) and DIERef::Decode(...)
  EncodeDecode(DIERef(std::nullopt, DIERef::Section::DebugInfo, 0x11223344));
  EncodeDecode(DIERef(std::nullopt, DIERef::Section::DebugTypes, 0x11223344));
  EncodeDecode(DIERef(100, DIERef::Section::DebugInfo, 0x11223344));
  EncodeDecode(DIERef(200, DIERef::Section::DebugTypes, 0x11223344));
}

static void EncodeDecode(const NameToDIE &object, ByteOrder byte_order) {
  const uint8_t addr_size = 8;
  DataEncoder encoder(byte_order, addr_size);
  DataEncoder strtab_encoder(byte_order, addr_size);
  ConstStringTable const_strtab;

  object.Encode(encoder, const_strtab);

  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);

  const_strtab.Encode(strtab_encoder);
  llvm::ArrayRef<uint8_t> strtab_bytes = strtab_encoder.GetData();
  DataExtractor strtab_data(strtab_bytes.data(), strtab_bytes.size(),
                            byte_order, addr_size);
  StringTableReader strtab_reader;
  offset_t strtab_data_offset = 0;
  ASSERT_EQ(strtab_reader.Decode(strtab_data, &strtab_data_offset), true);

  NameToDIE decoded_object;
  offset_t data_offset = 0;
  decoded_object.Decode(data, &data_offset, strtab_reader);
  EXPECT_EQ(object, decoded_object);
}

static void EncodeDecode(const NameToDIE &object) {
  EncodeDecode(object, eByteOrderLittle);
  EncodeDecode(object, eByteOrderBig);
}

TEST(DWARFIndexCachingTest, NameToDIEEncodeDecode) {
  NameToDIE map;
  // Make sure an empty NameToDIE map encodes and decodes correctly.
  EncodeDecode(map);
  map.Insert(ConstString("hello"),
             DIERef(std::nullopt, DIERef::Section::DebugInfo, 0x11223344));
  map.Insert(ConstString("workd"),
             DIERef(100, DIERef::Section::DebugInfo, 0x11223344));
  map.Finalize();
  // Make sure a valid NameToDIE map encodes and decodes correctly.
  EncodeDecode(map);
}

static void EncodeDecode(const ManualDWARFIndex::IndexSet &object,
                         ByteOrder byte_order) {
  const uint8_t addr_size = 8;
  DataEncoder encoder(byte_order, addr_size);
  DataEncoder strtab_encoder(byte_order, addr_size);
  object.Encode(encoder);
  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);
  ManualDWARFIndex::IndexSet decoded_object;
  offset_t data_offset = 0;
  decoded_object.Decode(data, &data_offset);
  EXPECT_TRUE(object == decoded_object);
}

static void EncodeDecode(const ManualDWARFIndex::IndexSet &object) {
  EncodeDecode(object, eByteOrderLittle);
  EncodeDecode(object, eByteOrderBig);
}

TEST(DWARFIndexCachingTest, ManualDWARFIndexIndexSetEncodeDecode) {
  ManualDWARFIndex::IndexSet set;
  // Make sure empty IndexSet can be encoded and decoded correctly
  EncodeDecode(set);

  dw_offset_t die_offset = 0;
  // Make sure an IndexSet with only items in IndexSet::function_basenames can
  // be encoded and decoded correctly.
  set.function_basenames.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.function_basenames.Clear();
  // Make sure an IndexSet with only items in IndexSet::function_fullnames can
  // be encoded and decoded correctly.
  set.function_fullnames.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.function_fullnames.Clear();
  // Make sure an IndexSet with only items in IndexSet::function_methods can
  // be encoded and decoded correctly.
  set.function_methods.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.function_methods.Clear();
  // Make sure an IndexSet with only items in IndexSet::function_selectors can
  // be encoded and decoded correctly.
  set.function_selectors.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.function_selectors.Clear();
  // Make sure an IndexSet with only items in IndexSet::objc_class_selectors can
  // be encoded and decoded correctly.
  set.objc_class_selectors.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.objc_class_selectors.Clear();
  // Make sure an IndexSet with only items in IndexSet::globals can
  // be encoded and decoded correctly.
  set.globals.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.globals.Clear();
  // Make sure an IndexSet with only items in IndexSet::types can
  // be encoded and decoded correctly.
  set.types.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.types.Clear();
  // Make sure an IndexSet with only items in IndexSet::namespaces can
  // be encoded and decoded correctly.
  set.namespaces.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
  set.namespaces.Clear();
  // Make sure that an IndexSet with item in all NameToDIE maps can be
  // be encoded and decoded correctly.
  set.function_basenames.Insert(
      ConstString("a"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.function_fullnames.Insert(
      ConstString("b"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.function_methods.Insert(
      ConstString("c"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.function_selectors.Insert(
      ConstString("d"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.objc_class_selectors.Insert(
      ConstString("e"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.globals.Insert(
      ConstString("f"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.types.Insert(
      ConstString("g"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  set.namespaces.Insert(
      ConstString("h"),
      DIERef(std::nullopt, DIERef::Section::DebugInfo, ++die_offset));
  EncodeDecode(set);
}

static void EncodeDecode(const CacheSignature &object, ByteOrder byte_order,
                         bool encode_result) {
  const uint8_t addr_size = 8;
  DataEncoder encoder(byte_order, addr_size);
  EXPECT_EQ(encode_result, object.Encode(encoder));
  if (!encode_result)
    return;
  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);
  offset_t data_offset = 0;
  CacheSignature decoded_object;
  EXPECT_TRUE(decoded_object.Decode(data, &data_offset));
  EXPECT_EQ(object, decoded_object);
}

static void EncodeDecode(const CacheSignature &object, bool encode_result) {
  EncodeDecode(object, eByteOrderLittle, encode_result);
  EncodeDecode(object, eByteOrderBig, encode_result);
}

TEST(DWARFIndexCachingTest, CacheSignatureTests) {
  CacheSignature sig;
  // A cache signature is only considered valid if it has a UUID.
  sig.m_mod_time = 0x12345678;
  EXPECT_FALSE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/false);
  sig.Clear();

  sig.m_obj_mod_time = 0x12345678;
  EXPECT_FALSE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/false);
  sig.Clear();

  sig.m_uuid = UUID("@\x00\x11\x22\x33\x44\x55\x66\x77", 8);
  EXPECT_TRUE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/true);
  sig.m_mod_time = 0x12345678;
  EXPECT_TRUE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/true);
  sig.m_obj_mod_time = 0x456789ab;
  EXPECT_TRUE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/true);
  sig.m_mod_time = std::nullopt;
  EXPECT_TRUE(sig.IsValid());
  EncodeDecode(sig, /*encode_result=*/true);

  // Recent changes do not allow cache signatures with only a modification time
  // or object modification time, so make sure if we try to decode such a cache
  // file that we fail. This verifies that if we try to load an previously
  // valid cache file where the signature is insufficient, that we will fail to
  // decode and load these cache files.
  DataEncoder encoder(eByteOrderLittle, /*addr_size=*/8);
  encoder.AppendU8(2); // eSignatureModTime
  encoder.AppendU32(0x12345678);
  encoder.AppendU8(255); // eSignatureEnd

  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  DataExtractor data(bytes.data(), bytes.size(), eByteOrderLittle,
                     /*addr_size=*/8);
  offset_t data_offset = 0;

  // Make sure we fail to decode a CacheSignature with only a mod time
  EXPECT_FALSE(sig.Decode(data, &data_offset));

  // Change the signature data to contain only a eSignatureObjectModTime and
  // make sure decoding fails as well.
  encoder.PutU8(/*offset=*/0, 3); // eSignatureObjectModTime
  data_offset = 0;
  EXPECT_FALSE(sig.Decode(data, &data_offset));

}
