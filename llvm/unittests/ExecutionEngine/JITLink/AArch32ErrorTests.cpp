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
orc::ExecutorAddr B1DummyAddr(0x1000);

auto ArmCfg = getArmConfigForCPUArch(ARMBuildAttrs::v7);

TEST(AArch32_ELF, readAddendErrors) {
  // Permanently undefined instruction
  //    11110:op:imm4:1:op1:imm12
  //    op  = 1111111 Permanent undefined
  //    op1 = 010
  ArrayRef<char> Content = "0xf7f0a000";
  constexpr uint64_t Alignment = 4;
  constexpr uint64_t AlignmentOffset = 0;
  auto &B = G->createContentBlock(Sec, Content, B1DummyAddr, Alignment,
                                  AlignmentOffset);
  constexpr orc::ExecutorAddrDiff Offset = 0;
  constexpr orc::ExecutorAddrDiff Size = 4;
  Symbol &TargetSymbol = G->addAnonymousSymbol(B, Offset, Size, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  EXPECT_THAT_EXPECTED(readAddendData(*G, B, InvalidEdge),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  EXPECT_THAT_EXPECTED(readAddendArm(*G, B, InvalidEdge),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  EXPECT_THAT_EXPECTED(readAddendThumb(*G, B, InvalidEdge, ArmCfg),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(
        readAddendArm(*G, B, E),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(
        readAddendThumb(*G, B, E, ArmCfg),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
}

TEST(AArch32_ELF, applyFixupErrors) {
  // Permanently undefined instruction
  char ContentArray[] = "0xf7f0a000";
  MutableArrayRef<char> MutableContent(ContentArray);
  constexpr uint64_t Alignment = 4;
  constexpr uint64_t AlignmentOffset = 0;
  auto &B = G->createMutableContentBlock(Sec, MutableContent, B1DummyAddr,
                                         Alignment, AlignmentOffset);

  constexpr orc::ExecutorAddrDiff Offset = 0;
  constexpr orc::ExecutorAddrDiff Size = 4;
  Symbol &TargetSymbol = G->addAnonymousSymbol(B, Offset, Size, false, false);
  Edge InvalidEdge(Edge::GenericEdgeKind::Invalid, 0 /*Offset*/, TargetSymbol,
                   0 /*Addend*/);

  EXPECT_THAT_ERROR(
      applyFixupData(*G, B, InvalidEdge),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));
  EXPECT_THAT_ERROR(
      applyFixupArm(*G, B, InvalidEdge),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));
  EXPECT_THAT_ERROR(
      applyFixupThumb(*G, B, InvalidEdge, ArmCfg),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixupArm(*G, B, E),
                      FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixupThumb(*G, B, E, ArmCfg),
                      FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
}
