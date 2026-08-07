//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for ContiguousBlobAccumulator.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/ContiguousBlobAccumulator.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::yaml;

static std::string getData(const ContiguousBlobAccumulator &CBA) {
  SmallString<16> Data;
  raw_svector_ostream OS(Data);
  CBA.writeBlobToStream(OS);
  return std::string(OS.str());
}

TEST(ContiguousBlobAccumulatorTest, Normal) {
  ContiguousBlobAccumulator CBA(/*BaseOffset=*/4, /*SizeLimit=*/64);
  EXPECT_EQ(CBA.tell(), 0u);
  EXPECT_EQ(CBA.getOffset(), 4u);

  raw_ostream *OS = CBA.getRawOS(1);
  ASSERT_NE(OS, nullptr);
  *OS << 'a';
  CBA.write("bc", 2);
  CBA.write(static_cast<unsigned char>('d'));
  CBA.writeZeros(2);
  CBA.write("12", 2);
  CBA.write(static_cast<uint16_t>(0x1234), llvm::endianness::big);
  BinaryRef Bin("333435");
  CBA.writeAsBinary(Bin, 2);

  EXPECT_EQ(CBA.padToAlignment(8), 16u);
  EXPECT_EQ(CBA.writeULEB128(0x7f), 1u);
  EXPECT_EQ(CBA.writeSLEB128(-1), 1u);

  const char ExpectedBeforeUpdate[] = {'a', 'b',  'c',  'd',  '\0', '\0', '1',
                                       '2', 0x12, 0x34, 0x33, 0x34, 0x7f, 0x7f};
  EXPECT_EQ(getData(CBA),
            std::string(ExpectedBeforeUpdate, sizeof(ExpectedBeforeUpdate)));

  CBA.updateDataAt(/*Pos=*/4, "ABCD", 4);
  CBA.updateDataAt(/*Pos=*/10, static_cast<uint16_t>(0x3536),
                   llvm::endianness::big);
  CBA.updateDataAt(/*Pos=*/12, static_cast<uint16_t>(0x3837),
                   llvm::endianness::little);

  const char ExpectedAfterUpdate[] = {'A',  'B',  'C',  'D',  '\0', '\0', 0x35,
                                      0x36, 0x37, 0x38, 0x33, 0x34, 0x7f, 0x7f};
  EXPECT_EQ(CBA.tell(), sizeof(ExpectedAfterUpdate));
  EXPECT_EQ(CBA.getOffset(), 4 + sizeof(ExpectedAfterUpdate));
  EXPECT_EQ(getData(CBA),
            std::string(ExpectedAfterUpdate, sizeof(ExpectedAfterUpdate)));
  EXPECT_THAT_ERROR(CBA.takeLimitError(), Succeeded());
}

TEST(ContiguousBlobAccumulatorTest, Invalid) {
  ContiguousBlobAccumulator CBA(/*BaseOffset=*/0, /*SizeLimit=*/4);
  CBA.write("abcd", 4);

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(CBA.updateDataAt(/*Pos=*/4, "Z", 1),
               "update range is invalid without reaching the output size "
               "limit");
#endif

  CBA.write("e", 1);
  CBA.updateDataAt(/*Pos=*/1, "XY", 2);
  CBA.updateDataAt(/*Pos=*/4, "Z", 1);
  EXPECT_EQ(getData(CBA), "aXYd");
  EXPECT_THAT_ERROR(CBA.takeLimitError(),
                    FailedWithMessage("reached the output size limit"));
}
