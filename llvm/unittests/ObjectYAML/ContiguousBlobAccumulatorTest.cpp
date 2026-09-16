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
  constexpr uint64_t BaseOffset = 4;
  ContiguousBlobAccumulator CBA(BaseOffset, /*SizeLimit=*/64);
  EXPECT_EQ(CBA.tell(), 0u);
  EXPECT_EQ(CBA.getOffset(), BaseOffset);

  raw_ostream *OS = CBA.getRawOS(1);
  ASSERT_NE(OS, nullptr);
  *OS << 'a';
  CBA.write("bc", 2);
  CBA.write(static_cast<unsigned char>('d'));
  CBA.writeZeros(1);
  CBA.write(static_cast<uint16_t>(0x1234), llvm::endianness::big);
  CBA.write(static_cast<uint16_t>(0x5678), llvm::endianness::little);
  BinaryRef Bin("333435");
  CBA.writeAsBinary(Bin, 2);

  EXPECT_EQ(CBA.tell(), 11u);
  EXPECT_EQ(CBA.padToAlignment(8), 16u);
  EXPECT_EQ(CBA.tell(), 12u);
  EXPECT_EQ(CBA.writeULEB128(0x7f), 1u);
  EXPECT_EQ(CBA.writeSLEB128(-2), 1u);

  const std::string ExpectedBeforeUpdate = {'a',  'b',  'c',  'd',  '\0',
                                            0x12, 0x34, 0x78, 0x56, 0x33,
                                            0x34, '\0', 0x7f, 0x7e};
  EXPECT_EQ(getData(CBA), ExpectedBeforeUpdate);

  CBA.updateDataAt(/*Pos=*/BaseOffset, "ABCD", 4);
  CBA.updateDataAt(/*Pos=*/BaseOffset + 5, static_cast<uint16_t>(0x3536),
                   llvm::endianness::little);
  CBA.updateDataAt(/*Pos=*/BaseOffset + 7, static_cast<uint16_t>(0x3738),
                   llvm::endianness::big);

  const std::string ExpectedAfterUpdate = {'A',  'B',  'C',  'D',  '\0',
                                           0x36, 0x35, 0x37, 0x38, 0x33,
                                           0x34, '\0', 0x7f, 0x7e};
  EXPECT_EQ(CBA.getOffset(), BaseOffset + ExpectedAfterUpdate.size());
  EXPECT_EQ(getData(CBA), ExpectedAfterUpdate);
  EXPECT_THAT_ERROR(CBA.takeLimitError(), Succeeded());
}

TEST(ContiguousBlobAccumulatorTest, ReachedLimit) {
  constexpr uint64_t BaseOffset = 0, SizeLimit = 4;
  ContiguousBlobAccumulator CBA(BaseOffset, SizeLimit);
  CBA.write("abcd", SizeLimit);
  EXPECT_EQ(CBA.getOffset(), BaseOffset + SizeLimit);
  // Trigger the size limit error so the out-of-bounds update below is ignored.
  CBA.write("e", 1);
  CBA.updateDataAt(/*Pos=*/BaseOffset + 1, "XY", 2);
  CBA.updateDataAt(/*Pos=*/BaseOffset + SizeLimit, "Z", 1);
  EXPECT_EQ(getData(CBA), "aXYd");
  EXPECT_THAT_ERROR(CBA.takeLimitError(),
                    FailedWithMessage("reached the output size limit"));
}
