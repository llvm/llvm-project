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


  EXPECT_THAT_EXPECTED(
      readAddend(*G, B, InvalidEdge, ArmCfg),
      FailedWithMessage(testing::HasSubstr("can not read implicit addend for aarch32 edge kind Keep-Alive")));

  std::string ReadAddendError = "can not read implicit addend for aarch32 edge kind <Unrecognized edge kind>";

  for (Edge::Kind K = FirstDataRelocation; K < LastDataRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E), Succeeded());
    EXPECT_THAT_EXPECTED(readAddendArm(*G, B, E),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
    EXPECT_THAT_EXPECTED(readAddendThumb(*G, B, E, ArmCfg),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
  }
  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
    EXPECT_THAT_EXPECTED(
        readAddendArm(*G, B, E),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
    EXPECT_THAT_EXPECTED(readAddendThumb(*G, B, E, ArmCfg),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
  }
  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_EXPECTED(readAddendData(*G, B, E),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
    EXPECT_THAT_EXPECTED(readAddendArm(*G, B, E),
                         FailedWithMessage(testing::HasSubstr(ReadAddendError)));
    EXPECT_THAT_EXPECTED(
        readAddendThumb(*G, B, E, ArmCfg),
        FailedWithMessage(testing::StartsWith("Invalid opcode")));
  }
}
