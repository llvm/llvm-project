//===- lib/MC/MCInstrInfo.cpp - Target Instruction Info -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

using namespace llvm;

const uint16_t *MCInstrNameTable::decodeBlock(unsigned Block) const {
  assert(Block < BlockOffsets.size());

  const uint8_t *ReadPtr = CompressedData.begin() + BlockOffsets[Block];
  [[maybe_unused]] const uint8_t *BlockEnd =
      Block + 1 == BlockOffsets.size()
          ? CompressedData.end()
          : CompressedData.begin() + BlockOffsets[Block + 1];
  unsigned NumBlockNames = std::min(BlockSize, NumNames - Block * BlockSize);
  assert(BlockEnd - ReadPtr >= 2);
  uint16_t DataSize = ReadPtr[0] | (uint16_t(ReadPtr[1]) << 8);
  ReadPtr += 2;

  size_t IndexBytes = (sizeof(uint16_t) + sizeof(uint8_t)) * NumBlockNames;
  void *Storage = ::operator new(IndexBytes + DataSize);
  auto *Indices = static_cast<uint16_t *>(Storage);
  auto *Lengths = reinterpret_cast<uint8_t *>(Indices + NumBlockNames);
  char *Data = reinterpret_cast<char *>(Lengths + NumBlockNames);
  char *WritePtr = Data;
  StringRef Previous;

  for (unsigned I = 0; I != NumBlockNames; ++I) {
    assert(BlockEnd - ReadPtr >= 2);
    uint8_t PrefixLength = *ReadPtr++;
    uint8_t SuffixLength = *ReadPtr++;
    assert(PrefixLength <= Previous.size());
    assert(SuffixLength <= BlockEnd - ReadPtr);

    char *NameStart = WritePtr;
    if (PrefixLength) {
      std::memcpy(WritePtr, Previous.data(), PrefixLength);
      WritePtr += PrefixLength;
    }
    if (SuffixLength) {
      std::memcpy(WritePtr, ReadPtr, SuffixLength);
      WritePtr += SuffixLength;
      ReadPtr += SuffixLength;
    }
    *WritePtr++ = '\0';

    assert(NameStart - Data <= std::numeric_limits<uint16_t>::max());
    Indices[I] = static_cast<uint16_t>(NameStart - Data);
    Lengths[I] = PrefixLength + SuffixLength;
    Previous = StringRef(NameStart, Lengths[I]);
  }

  assert(ReadPtr == BlockEnd);
  assert(WritePtr == Data + DataSize);

  const uint16_t *Expected = nullptr;
  if (!DecodedBlocks[Block].compare_exchange_strong(
          Expected, Indices, std::memory_order_release,
          std::memory_order_acquire)) {
    ::operator delete(Storage);
    return Expected;
  }
  return Indices;
}

bool MCInstrInfo::getDeprecatedInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                    std::string &Info) const {
  unsigned Opcode = MI.getOpcode();
  if (ComplexDeprecationInfos && ComplexDeprecationInfos[Opcode])
    return ComplexDeprecationInfos[Opcode](MI, STI, Info);
  if (DeprecatedFeatures && DeprecatedFeatures[Opcode] != uint8_t(-1U) &&
      STI.getFeatureBits()[DeprecatedFeatures[Opcode]]) {
    // FIXME: it would be nice to include the subtarget feature here.
    Info = "deprecated";
    return true;
  }
  return false;
}
