//===- llvm/unittest/Target/AMDGPU/BlockingCycles.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Test that GFX1250 blocking cycles entries are defined for the expected
/// instructions and with the expected values.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "gtest/gtest.h"

#include "llvm/MC/MCInstrInfo.h"

namespace {
using namespace llvm;
using namespace llvm::AMDGPU;

struct AMDGPUBlockingCyclesInfo {
  uint16_t Opcode;
  uint8_t GFX1250BlockingCycles;
};

#define GET_GFX1250BlockingCyclesTable_DECL
#define GET_GFX1250BlockingCyclesTable_IMPL
#include "AMDGPUGenSearchableTables.inc"
} // anonymous namespace

class BlockingCyclesTest : public AMDGPUTestBase {};

TEST_F(BlockingCyclesTest, GFX1250EntryCounts) {
  std::unique_ptr<GCNTargetMachine> TM =
      createAMDGPUTargetMachine(Triple("amdgpu-amd-amdhsa"), "gfx1250", "");
  const MCInstrInfo *MCII = TM->getMCInstrInfo();

  unsigned Total = 0;
  unsigned Cycles2 = 0, Cycles4 = 0, Cycles8 = 0, Cycles16 = 0, Cycles32 = 0;

  for (unsigned Op = 0; Op < MCII->getNumOpcodes(); ++Op) {
    const auto *Entry = getGFX1250BlockingCyclesInfo(Op);
    if (!Entry)
      continue;

    ++Total;
    switch (Entry->GFX1250BlockingCycles) {
    case 2:
      ++Cycles2;
      break;
    case 4:
      ++Cycles4;
      break;
    case 8:
      ++Cycles8;
      break;
    case 16:
      ++Cycles16;
      break;
    case 32:
      ++Cycles32;
      break;
    default:
      FAIL() << "Unexpected blocking cycles value "
             << (unsigned)Entry->GFX1250BlockingCycles << " for "
             << MCII->getName(Op);
    }
  }

  EXPECT_EQ(Total, 2543u);
  EXPECT_EQ(Cycles2, 2223u);
  EXPECT_EQ(Cycles4, 53u);
  EXPECT_EQ(Cycles8, 20u);
  EXPECT_EQ(Cycles16, 16u);
  EXPECT_EQ(Cycles32, 231u);
}
