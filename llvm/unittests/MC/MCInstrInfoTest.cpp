//===- MCInstrInfoTest.cpp - MCInstrInfo unit tests
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Config/llvm-config.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <new>
#include <string>
#include <thread>
#include <vector>

using namespace llvm;

namespace {

TEST(MCInstrInfoTest, FrontCodedInstructionNames) {
  constexpr unsigned NumNames = MCInstrNameTable::BlockSize + 2;
  constexpr unsigned NumBlocks = 2;

  std::vector<std::string> Names;
  Names.reserve(NumNames);
  for (unsigned I = 0; I != NumNames; ++I)
    Names.push_back("COMMON_INSTRUCTION_PREFIX_" + std::to_string(I));

  std::vector<uint8_t> CompressedData;
  std::vector<uint32_t> BlockOffsets;

  for (unsigned Start = 0; Start < NumNames;
       Start += MCInstrNameTable::BlockSize) {
    unsigned End = std::min(Start + MCInstrNameTable::BlockSize, NumNames);
    BlockOffsets.push_back(CompressedData.size());

    unsigned DataSize = 0;
    for (unsigned I = Start; I != End; ++I)
      DataSize += Names[I].size() + 1;
    CompressedData.push_back(DataSize);
    CompressedData.push_back(DataSize >> 8);

    StringRef Previous;
    for (unsigned I = Start; I != End; ++I) {
      StringRef Name = Names[I];
      unsigned PrefixLength = 0;
      while (PrefixLength != Previous.size() && PrefixLength != Name.size() &&
             Previous[PrefixLength] == Name[PrefixLength])
        ++PrefixLength;
      CompressedData.push_back(PrefixLength);
      CompressedData.push_back(Name.size() - PrefixLength);
      CompressedData.insert(CompressedData.end(),
                            Name.bytes_begin() + PrefixLength,
                            Name.bytes_end());
      Previous = Name;
    }
  }

  std::array<std::atomic<const uint16_t *>, NumBlocks> DecodedBlocks;
  for (auto &Block : DecodedBlocks)
    Block.store(nullptr, std::memory_order_relaxed);
  MCInstrNameTable Table(CompressedData, BlockOffsets, DecodedBlocks.data(),
                         NumNames);

#if LLVM_ENABLE_THREADS
  std::atomic<bool> Correct{true};
  std::array<std::thread, 8> Threads;
  for (auto &Thread : Threads)
    Thread = std::thread([&] {
      for (unsigned I = 0; I != NumNames; ++I)
        if (Table.getName(I) != Names[I])
          Correct.store(false, std::memory_order_relaxed);
    });
  for (auto &Thread : Threads)
    Thread.join();
  EXPECT_TRUE(Correct.load(std::memory_order_relaxed));
#endif

  for (unsigned I = 0; I != NumNames; ++I)
    EXPECT_EQ(Table.getName(I), Names[I]);

  for (auto &Block : DecodedBlocks)
    ::operator delete(const_cast<uint16_t *>(Block.load()));
}

} // namespace
