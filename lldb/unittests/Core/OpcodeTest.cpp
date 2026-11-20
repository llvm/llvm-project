//===-- OpcodeTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Opcode.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"
#include <vector>

using namespace lldb;
using namespace lldb_private;

TEST(OpcodeTest, InlineOpcodeBytes) {
  // Test 1 byte
  uint8_t b1[] = {0xaa};
  Opcode op1;
  op1.SetOpcodeBytes(b1, sizeof(b1));
  EXPECT_EQ(op1.GetType(), Opcode::eTypeBytes);
  EXPECT_EQ(op1.GetByteSize(), 1u);
  EXPECT_EQ(memcmp(op1.GetOpcodeBytes(), b1, 1), 0);

  // Test 8 bytes
  uint8_t b8[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  Opcode op8;
  op8.SetOpcodeBytes(b8, sizeof(b8));
  EXPECT_EQ(op8.GetType(), Opcode::eTypeBytes);
  EXPECT_EQ(op8.GetByteSize(), 8u);
  EXPECT_EQ(memcmp(op8.GetOpcodeBytes(), b8, 8), 0);

  // Test 16 bytes (kMaxByteSize inline capacity)
  uint8_t b16[16];
  for (size_t i = 0; i < sizeof(b16); ++i)
    b16[i] = static_cast<uint8_t>(i);
  Opcode op16;
  op16.SetOpcodeBytes(b16, sizeof(b16));
  EXPECT_EQ(op16.GetType(), Opcode::eTypeBytes);
  EXPECT_EQ(op16.GetByteSize(), 16u);
  EXPECT_EQ(memcmp(op16.GetOpcodeBytes(), b16, 16), 0);
}

TEST(OpcodeTest, OversizedOpcodeBytes) {
  for (size_t size : {32, 64, 128}) {
    std::vector<uint8_t> bytes(size);
    for (size_t i = 0; i < size; ++i)
      bytes[i] = static_cast<uint8_t>(i ^ 0x5a);

    Opcode op;
    op.SetOpcodeBytes(bytes.data(), bytes.size());
    EXPECT_EQ(op.GetType(), Opcode::eTypeBytes);
    EXPECT_EQ(op.GetByteSize(), size);
    EXPECT_NE(op.GetOpcodeBytes(), nullptr);
    EXPECT_EQ(memcmp(op.GetOpcodeBytes(), bytes.data(), size), 0);

    DataExtractor extractor;
    EXPECT_EQ(op.GetData(extractor), size);
    EXPECT_EQ(extractor.GetByteSize(), size);
    EXPECT_EQ(memcmp(extractor.GetDataStart(), bytes.data(), size), 0);
  }
}

TEST(OpcodeTest, Thumb16_2ByteOrdering) {
  uint32_t thumb_inst = 0xe92d41f0;

  Opcode op_le;
  op_le.SetOpcode16_2(thumb_inst, eByteOrderLittle);
  EXPECT_EQ(op_le.GetType(), Opcode::eType16_2);
  EXPECT_EQ(op_le.GetByteSize(), 4u);

  DataExtractor extractor_le;
  EXPECT_EQ(op_le.GetData(extractor_le), 4u);
  offset_t offset = 0;
  uint16_t hw1 = extractor_le.GetU16(&offset);
  uint16_t hw2 = extractor_le.GetU16(&offset);
  EXPECT_EQ(hw1, 0xe92d);
  EXPECT_EQ(hw2, 0x41f0);

  Opcode op_be;
  op_be.SetOpcode16_2(thumb_inst, eByteOrderBig);
  EXPECT_EQ(op_be.GetType(), Opcode::eType16_2);
  EXPECT_EQ(op_be.GetByteSize(), 4u);
  DataExtractor extractor_be;
  EXPECT_EQ(op_be.GetData(extractor_be), 4u);
}

TEST(OpcodeTest, DumpBytesAndTuples) {
  // Dump for eTypeBytes (both inline and > 16B)
  uint8_t b4[] = {0x12, 0x34, 0x56, 0x78};
  Opcode op4;
  op4.SetOpcodeBytes(b4, sizeof(b4));
  StreamString ss;
  op4.Dump(&ss, 0);
  EXPECT_EQ(ss.GetString(), "12 34 56 78");

  std::vector<uint8_t> b24(24);
  for (size_t i = 0; i < 24; ++i)
    b24[i] = static_cast<uint8_t>(i);
  Opcode op24;
  op24.SetOpcodeBytes(b24.data(), b24.size());
  ss.Clear();
  op24.Dump(&ss, 0);
  EXPECT_EQ(ss.GetString(), "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f "
                            "10 11 12 13 14 15 16 17");

  // Dump for eType16_32Tuples
  uint8_t b_tuple[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  Opcode op_tuple;
  op_tuple.SetOpcode16_32TupleBytes(b_tuple, sizeof(b_tuple), eByteOrderLittle);
  EXPECT_EQ(op_tuple.GetType(), Opcode::eType16_32Tuples);
  EXPECT_EQ(op_tuple.GetByteSize(), 8u);
  ss.Clear();
  op_tuple.Dump(&ss, 0);
  EXPECT_EQ(ss.GetString(), "04030201 08070605");
}

TEST(OpcodeTest, CopyAndMoveSemantics) {
  // Test inline opcode copy and move
  uint8_t inline_bytes[] = {0x11, 0x22, 0x33, 0x44};
  Opcode op_inline;
  op_inline.SetOpcodeBytes(inline_bytes, sizeof(inline_bytes));

  Opcode op_inline_copy(op_inline);
  EXPECT_EQ(op_inline_copy.GetByteSize(), 4u);
  EXPECT_EQ(memcmp(op_inline_copy.GetOpcodeBytes(), inline_bytes, 4), 0);

  Opcode op_inline_assign;
  op_inline_assign = op_inline;
  EXPECT_EQ(op_inline_assign.GetByteSize(), 4u);
  EXPECT_EQ(memcmp(op_inline_assign.GetOpcodeBytes(), inline_bytes, 4), 0);

  Opcode op_inline_move(std::move(op_inline));
  EXPECT_EQ(op_inline_move.GetByteSize(), 4u);
  EXPECT_EQ(memcmp(op_inline_move.GetOpcodeBytes(), inline_bytes, 4), 0);

  Opcode op_inline_move_assign;
  op_inline_move_assign = std::move(op_inline_copy);
  EXPECT_EQ(op_inline_move_assign.GetByteSize(), 4u);
  EXPECT_EQ(memcmp(op_inline_move_assign.GetOpcodeBytes(), inline_bytes, 4), 0);

  // Test oversized heap-spilled opcode copy and move
  std::vector<uint8_t> over_bytes(48, 0x7f);
  Opcode op_over;
  op_over.SetOpcodeBytes(over_bytes.data(), over_bytes.size());
  EXPECT_EQ(op_over.GetByteSize(), 48u);

  Opcode op_over_copy(op_over);
  EXPECT_EQ(op_over_copy.GetByteSize(), 48u);
  EXPECT_EQ(memcmp(op_over_copy.GetOpcodeBytes(), over_bytes.data(), 48), 0);

  Opcode op_over_assign;
  op_over_assign = op_over;
  EXPECT_EQ(op_over_assign.GetByteSize(), 48u);
  EXPECT_EQ(memcmp(op_over_assign.GetOpcodeBytes(), over_bytes.data(), 48), 0);

  Opcode op_over_move(std::move(op_over));
  EXPECT_EQ(op_over_move.GetByteSize(), 48u);
  EXPECT_EQ(memcmp(op_over_move.GetOpcodeBytes(), over_bytes.data(), 48), 0);

  Opcode op_over_move_assign;
  op_over_move_assign = std::move(op_over_copy);
  EXPECT_EQ(op_over_move_assign.GetByteSize(), 48u);
  EXPECT_EQ(memcmp(op_over_move_assign.GetOpcodeBytes(), over_bytes.data(), 48),
            0);
}

TEST(OpcodeTest, UnionSizeAndLifetimeTransitions) {
  static_assert(sizeof(Opcode) <= 48,
                "SmallVector should share union storage with scalar opcodes");
  static_assert(sizeof(void *) != 8 || sizeof(Opcode) == 48,
                "Expected 48-byte Opcode on 64-bit targets");

  std::vector<uint8_t> over48(48, 0xa5);
  std::vector<uint8_t> over64(64, 0x3c);
  uint8_t inline8[] = {1, 2, 3, 4, 5, 6, 7, 8};

  Opcode op;
  // Heap-spilled bytes -> SetOpcode8 -> heap-spilled bytes -> SetOpcode16 ->
  // heap-spilled tuples -> SetOpcode16_2 -> inline bytes -> SetOpcode32 ->
  // heap-spilled bytes -> SetOpcode64 -> heap-spilled bytes -> Clear().
  op.SetOpcodeBytes(over48.data(), over48.size());
  EXPECT_EQ(op.GetType(), Opcode::eTypeBytes);
  EXPECT_EQ(op.GetByteSize(), 48u);

  op.SetOpcode8(0x7f, eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType8);
  EXPECT_EQ(op.GetOpcode8(), 0x7fu);
  EXPECT_EQ(op.GetOpcodeBytes(), nullptr);

  op.SetOpcodeBytes(over64.data(), over64.size());
  EXPECT_EQ(op.GetByteSize(), 64u);
  EXPECT_EQ(memcmp(op.GetOpcodeBytes(), over64.data(), 64), 0);

  op.SetOpcode16(0x1234, eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType16);
  EXPECT_EQ(op.GetOpcode16(), 0x1234u);

  op.SetOpcode16_32TupleBytes(over48.data(), over48.size(), eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType16_32Tuples);
  EXPECT_EQ(op.GetByteSize(), 48u);

  op.SetOpcode16_2(0xe92d41f0, eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType16_2);
  EXPECT_EQ(op.GetOpcode32(), 0xe92d41f0u);

  op.SetOpcodeBytes(inline8, sizeof(inline8));
  EXPECT_EQ(op.GetByteSize(), 8u);
  EXPECT_EQ(memcmp(op.GetOpcodeBytes(), inline8, 8), 0);

  op.SetOpcode32(0xdeadbeef, eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType32);
  EXPECT_EQ(op.GetOpcode32(), 0xdeadbeefu);

  op.SetOpcodeBytes(over48.data(), over48.size());
  op.SetOpcode64(0x0102030405060708ULL, eByteOrderLittle);
  EXPECT_EQ(op.GetType(), Opcode::eType64);
  EXPECT_EQ(op.GetOpcode64(), 0x0102030405060708ULL);

  op.SetOpcodeBytes(over64.data(), over64.size());
  op.SetOpcodeBytes(nullptr, 0);
  EXPECT_EQ(op.GetType(), Opcode::eTypeInvalid);
  EXPECT_EQ(op.GetByteSize(), 0u);
  EXPECT_EQ(op.GetOpcodeBytes(), nullptr);

  op.SetOpcodeBytes(over48.data(), over48.size());
  op.Clear();
  EXPECT_EQ(op.GetType(), Opcode::eTypeInvalid);
  EXPECT_EQ(op.GetByteSize(), 0u);

  // Cross-variant and byte-to-byte copy/move assignments.
  Opcode bytes_a;
  bytes_a.SetOpcodeBytes(over48.data(), over48.size());
  Opcode bytes_b;
  bytes_b.SetOpcode16_32TupleBytes(over64.data(), over64.size(),
                                   eByteOrderLittle);
  Opcode scalar32(uint32_t(0xcafebabe), eByteOrderLittle);
  Opcode invalid_op;

  // ByteBuf -> ByteBuf copy & move assignment.
  bytes_a = bytes_b;
  EXPECT_EQ(bytes_a.GetType(), Opcode::eType16_32Tuples);
  EXPECT_EQ(bytes_a.GetByteSize(), 64u);
  EXPECT_EQ(memcmp(bytes_a.GetOpcodeBytes(), over64.data(), 64), 0);

  // Self copy & move assignment via aliases.
  const Opcode &self_const = bytes_a;
  bytes_a = self_const;
  EXPECT_EQ(bytes_a.GetByteSize(), 64u);
  Opcode &self_mut = bytes_a;
  bytes_a = std::move(self_mut);
  EXPECT_EQ(bytes_a.GetByteSize(), 64u);

  // ByteBuf -> scalar -> ByteBuf -> invalid transitions via assignment.
  bytes_a = scalar32;
  EXPECT_EQ(bytes_a.GetType(), Opcode::eType32);
  EXPECT_EQ(bytes_a.GetOpcode32(), 0xcafebabeu);

  bytes_a = std::move(bytes_b);
  EXPECT_EQ(bytes_a.GetType(), Opcode::eType16_32Tuples);
  EXPECT_EQ(bytes_a.GetByteSize(), 64u);
  EXPECT_EQ(bytes_b.GetType(), Opcode::eTypeInvalid);

  bytes_a = invalid_op;
  EXPECT_EQ(bytes_a.GetType(), Opcode::eTypeInvalid);
}
