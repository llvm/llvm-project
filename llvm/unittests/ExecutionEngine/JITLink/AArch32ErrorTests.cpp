//===------- AArch32ErrorTests.cpp - Test AArch32 error handling ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ExecutionEngine/JITLink/aarch32.h>

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::aarch32;
using namespace llvm::support;
using namespace llvm::support::endian;

constexpr unsigned PointerSize = 4;
auto G = std::make_unique<LinkGraph>("foo", Triple("armv7-linux-gnueabi"),
                                     PointerSize, llvm::endianness::little,
                                     getGenericEdgeKindName);
auto &Sec =
    G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

auto ArmCfg = getArmConfigForCPUArch(ARMBuildAttrs::v7);

constexpr uint64_t ArmAlignment = 4;
constexpr uint64_t ThumbAlignment = 2;
constexpr uint64_t AlignmentOffset = 0;

constexpr orc::ExecutorAddrDiff SymbolOffset = 0;
constexpr orc::ExecutorAddrDiff SymbolSize = 4;

TEST(AArch32_ELF, readAddendArmErrors) {

  constexpr orc::ExecutorAddr B1DummyAddr(0x1000);

  // Permanently undefined instruction in ARM
  //    udf #0
  uint8_t ArmWord[] = {0xf0, 0x00, 0xf0, 0xe7};
  ArrayRef<char> ArmContent(reinterpret_cast<const char *>(&ArmWord),
                            sizeof(ArmWord));
  auto &BArm = G->createContentBlock(Sec, ArmContent, B1DummyAddr, ArmAlignment,
                                     AlignmentOffset);
  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BArm, SymbolOffset, SymbolSize, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  // Edge kind is tested, block itself is not significant here. So it is tested
  // once in Arm
  EXPECT_THAT_EXPECTED(readAddendData(*G, BArm, InvalidEdge),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  EXPECT_THAT_EXPECTED(readAddendArm(*G, BArm, InvalidEdge),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  EXPECT_THAT_EXPECTED(readAddendThumb(*G, BArm, InvalidEdge, ArmCfg),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(
        readAddendArm(*G, BArm, E),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
}

TEST(AArch32_ELF, readAddendThumbErrors) {

  constexpr orc::ExecutorAddr B2DummyAddr(0x2000);

  // Permanently undefined instruction in Thumb
  //    udf #0
  //
  //    11110:op:imm4:1:op1:imm12
  //    op  = 1111111 Permanent undefined
  //    op1 = 010
  //
  constexpr HalfWords ThumbHalfWords{0xf7f0, 0xa000};
  ArrayRef<char> ThumbContent(reinterpret_cast<const char *>(&ThumbHalfWords),
                              sizeof(ThumbHalfWords));
  auto &BThumb = G->createContentBlock(Sec, ThumbContent, B2DummyAddr,
                                       ThumbAlignment, AlignmentOffset);
  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BThumb, SymbolOffset, SymbolSize, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(
        readAddendThumb(*G, BThumb, E, ArmCfg),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
}

TEST(AArch32_ELF, applyFixupArmErrors) {

  constexpr orc::ExecutorAddr B3DummyAddr(0x5000);

  uint8_t ArmWord[] = {0xf0, 0x00, 0xf0, 0xe7};
  MutableArrayRef<char> MutableArmContent(reinterpret_cast<char *>(ArmWord),
                                          sizeof(ArmWord));

  auto &BArm = G->createMutableContentBlock(Sec, MutableArmContent, B3DummyAddr,
                                            ArmAlignment, AlignmentOffset);

  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BArm, SymbolOffset, SymbolSize, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  // Edge kind is tested, block itself is not significant here. So it is tested
  // once in Arm
  EXPECT_THAT_ERROR(
      applyFixupData(*G, BArm, InvalidEdge),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));
  EXPECT_THAT_ERROR(
      applyFixupArm(*G, BArm, InvalidEdge),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));
  EXPECT_THAT_ERROR(
      applyFixupThumb(*G, BArm, InvalidEdge, ArmCfg),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixupArm(*G, BArm, E),
                      FailedWithMessage(testing::AllOf(
                          testing::StartsWith("Invalid opcode"),
                          testing::EndsWith(G->getEdgeKindName(K)))));
  }
}

TEST(AArch32_ELF, applyFixupThumbErrors) {

  struct MutableHalfWords {
    constexpr MutableHalfWords(HalfWords Preset)
        : Hi(Preset.Hi), Lo(Preset.Lo) {}

    uint16_t Hi; // First halfword
    uint16_t Lo; // Second halfword
  };

  constexpr orc::ExecutorAddr B4DummyAddr(0x6000);

  // Permanently undefined instruction in Thumb
  //    udf #0
  //
  //    11110:op:imm4:1:op1:imm12
  //    op  = 1111111 Permanent undefined
  //    op1 = 010
  //
  constexpr HalfWords ThumbHalfWords{0xf7f0, 0xa000};
  MutableHalfWords MutableThumbHalfWords{ThumbHalfWords};
  MutableArrayRef<char> MutableThumbContent(
      reinterpret_cast<char *>(&MutableThumbHalfWords),
      sizeof(MutableThumbHalfWords));

  auto &BThumb = G->createMutableContentBlock(
      Sec, MutableThumbContent, B4DummyAddr, ThumbAlignment, AlignmentOffset);
  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BThumb, SymbolOffset, SymbolSize, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixupThumb(*G, BThumb, E, ArmCfg),
                      FailedWithMessage(testing::AllOf(
                          testing::StartsWith("Invalid opcode"),
                          testing::EndsWith(G->getEdgeKindName(K)))));
  }
}
