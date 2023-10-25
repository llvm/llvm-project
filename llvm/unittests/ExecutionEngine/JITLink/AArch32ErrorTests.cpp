//===------- AArch32ErrorTests.cpp - Unit tests for AArch32 error handling -===//
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

struct ThumbRelocation {
  /// Create a read-only reference to a Thumb32 fixup.
  ThumbRelocation(const char *FixupPtr)
      : Hi{*reinterpret_cast<const support::ulittle16_t *>(FixupPtr)},
        Lo{*reinterpret_cast<const support::ulittle16_t *>(FixupPtr + 2)} {}

  const support::ulittle16_t &Hi; // First halfword
  const support::ulittle16_t &Lo; // Second halfword
};

struct ArmRelocation {

  ArmRelocation(const char *FixupPtr)
      : Wd{*reinterpret_cast<const support::ulittle32_t *>(FixupPtr)} {}

  const support::ulittle32_t &Wd;
};

std::string makeUnexpectedOpcodeError(const LinkGraph &G,
                                      const ThumbRelocation &R,
                                      Edge::Kind Kind) {
  return formatv("Invalid opcode [ {0:x4}, {1:x4} ] for relocation: {2}",
                 static_cast<uint16_t>(R.Hi), static_cast<uint16_t>(R.Lo),
                 G.getEdgeKindName(Kind));
}

std::string makeUnexpectedOpcodeError(const LinkGraph &G,
                                      const ArmRelocation &R, Edge::Kind Kind) {
  return formatv("Invalid opcode {0:x8} for relocation: {1}",
                 static_cast<uint32_t>(R.Wd), G.getEdgeKindName(Kind));
}

TEST(AArch32_ELF, readAddends) {
  auto G = std::make_unique<LinkGraph>("foo", Triple("armv7-linux-gnueabi"), 4,
                                       llvm::endianness::little,
                                       getGenericEdgeKindName);

  ArrayRef<char> Content = "hello, world!";
  auto &Sec =
      G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B = G->createContentBlock(Sec, Content, B1Addr, 4, 0);

  Symbol &TargetSymbol = G->addAnonymousSymbol(B, 0, 4, false, false);
  Edge InvalidEdge(FirstDataRelocation - 1, 0, TargetSymbol, 0);

  auto ArmCfg = getArmConfigForCPUArch(ARMBuildAttrs::v7);

  auto makeReadAddendError = [](LinkGraph &G, Block &B, Edge &E) {
    return ("In graph " + G.getName() + ", section " +
            B.getSection().getName() +
            " can not read implicit addend for aarch32 edge kind " +
            G.getEdgeKindName(E.getKind()))
        .str();
  };

  EXPECT_THAT_EXPECTED(
      readAddend(*G, B, InvalidEdge, ArmCfg),
      FailedWithMessage(makeReadAddendError(*G, B, InvalidEdge)));

  ArmRelocation R_Arm(B.getContent().data());
  ThumbRelocation R_Thumb(B.getContent().data());

  for (Edge::Kind K = FirstDataRelocation; K < LastDataRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E), Succeeded());
    EXPECT_THAT_EXPECTED(readAddendArm(*G, B, E),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
    EXPECT_THAT_EXPECTED(readAddendThumb(*G, B, E, ArmCfg),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
  }
  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
    EXPECT_THAT_EXPECTED(
        readAddendArm(*G, B, E),
        FailedWithMessage(makeUnexpectedOpcodeError(*G, R_Arm, K)));
    EXPECT_THAT_EXPECTED(readAddendThumb(*G, B, E, ArmCfg),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
  }
  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
    EXPECT_THAT_EXPECTED(readAddendArm(*G, B, E),
                         FailedWithMessage(makeReadAddendError(*G, B, E)));
    EXPECT_THAT_EXPECTED(
        readAddendThumb(*G, B, E, ArmCfg),
        FailedWithMessage(makeUnexpectedOpcodeError(*G, R_Thumb, K)));
  }
}
