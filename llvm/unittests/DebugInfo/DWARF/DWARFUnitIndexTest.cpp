//===- llvm/unittest/DebugInfo/DWARF/DWARFUnitIndexTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <vector>

using namespace llvm;

namespace {

static void appendU32(std::vector<uint8_t> &B, uint32_t V) {
  uint8_t Tmp[4];
  support::endian::write32le(Tmp, V);
  B.insert(B.end(), Tmp, Tmp + 4);
}

static void appendU64(std::vector<uint8_t> &B, uint64_t V) {
  uint8_t Tmp[8];
  support::endian::write64le(Tmp, V);
  B.insert(B.end(), Tmp, Tmp + 8);
}

// A crafted .debug_cu_index (version 2) with one hash bucket whose parallel-
// table Index is NumUnits + 1. Without the bounds check parseImpl() writes
// Contribs[Index - 1], one slot past a heap array sized to NumUnits. parse()
// must reject it and leave the index empty.
TEST(DWARFUnitIndexTest, RejectsRowIndexBeyondNumUnits) {
  std::vector<uint8_t> Buffer;
  // Header: Version=2, NumColumns=1, NumUnits=1, NumBuckets=1.
  appendU32(Buffer, 2);
  appendU32(Buffer, 1);
  appendU32(Buffer, 1);
  appendU32(Buffer, 1);
  // Hash table of signatures: NumBuckets x u64.
  appendU64(Buffer, 0xdeadbeefULL);
  // Parallel table of indexes: NumBuckets x u32. Index = NumUnits + 1 = 2.
  appendU32(Buffer, 2);
  // Column headers: NumColumns x u32 (DW_SECT_INFO serializes to 1 under v2).
  appendU32(Buffer, 1);
  // Table of section offsets: NumUnits*NumColumns x u32.
  appendU32(Buffer, 0);
  // Table of section sizes: NumUnits*NumColumns x u32.
  appendU32(Buffer, 0);

  DataExtractor Data(
      StringRef(reinterpret_cast<const char *>(Buffer.data()), Buffer.size()),
      /*IsLittleEndian=*/true);
  DWARFUnitIndex Index(DW_SECT_INFO);
  EXPECT_FALSE(Index.parse(Data));
  // On failure parse() resets the header, so the index is falsy.
  EXPECT_FALSE(static_cast<bool>(Index));
}

} // end anonymous namespace
