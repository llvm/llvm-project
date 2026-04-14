//===---------- MachOBuilderTest.cpp - MachOBuilder Tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachOBuilder.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

static Expected<std::unique_ptr<object::MachOObjectFile>>
parseMachO(ArrayRef<char> Buffer) {
  return object::ObjectFile::createMachOObjectFile(
      MemoryBufferRef(StringRef(Buffer.data(), Buffer.size()), "test"));
}

TEST(MachOBuilderTest, AddLCUUID) {
  MachOBuilder<MachO64LE> B(4096);
  B.Header.filetype = MachO::MH_OBJECT;

  uint8_t UUID[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

  B.addLoadCommand<MachO::LC_UUID>(UUID);

  size_t Size = B.layout();
  std::vector<char> Buffer(Size, 0);
  B.write({Buffer.data(), Buffer.size()});

  auto Obj = parseMachO(Buffer);
  ASSERT_THAT_EXPECTED(Obj, Succeeded());

  ArrayRef<uint8_t> ParsedUUID = (*Obj)->getUuid();
  ASSERT_EQ(ParsedUUID.size(), 16u);
  EXPECT_EQ(ArrayRef<uint8_t>(UUID), ParsedUUID);
}
