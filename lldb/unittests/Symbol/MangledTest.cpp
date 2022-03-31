//===-- SymbolTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Mangled.h"
#include "lldb/Core/DataFileCache.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

static void EncodeDecode(const Mangled &object, ByteOrder byte_order) {
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

  Mangled decoded_object;
  offset_t data_offset = 0;
  decoded_object.Decode(data, &data_offset, strtab_reader);
  EXPECT_EQ(object, decoded_object);
}

static void EncodeDecode(const Mangled &object) {
  EncodeDecode(object, eByteOrderLittle);
  EncodeDecode(object, eByteOrderBig);
}

TEST(MangledTest, EncodeDecodeMangled) {
  Mangled mangled;
  // Test encoding and decoding an empty mangled object.
  EncodeDecode(mangled);

  // Test encoding a mangled object that hasn't demangled its name yet.
  mangled.SetMangledName(ConstString("_Z3fooi"));
  EncodeDecode(mangled);

  // Test encoding a mangled object that has demangled its name by computing it.
  mangled.GetDemangledName();
  // EncodeDecode(mangled);

  // Test encoding a mangled object that has just a demangled name
  mangled.SetMangledName(ConstString());
  mangled.SetDemangledName(ConstString("hello"));
  EncodeDecode(mangled);

  // Test encoding a mangled name that has both a mangled and demangled name
  // that are not mangled/demangled counterparts of each other.
  mangled.SetMangledName(ConstString("world"));
  EncodeDecode(mangled);
}
