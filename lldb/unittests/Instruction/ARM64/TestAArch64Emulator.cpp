//===-- TestAArch64Emulator.cpp ------------------------------------------===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"

using namespace lldb;
using namespace lldb_private;

struct Arch64EmulatorTester : public EmulateInstructionARM64 {
  Arch64EmulatorTester()
      : EmulateInstructionARM64(ArchSpec("arm64-apple-ios")) {}

  static uint64_t AddWithCarry(uint32_t N, uint64_t x, uint64_t y, bool carry_in,
                               EmulateInstructionARM64::ProcState &proc_state) {
    return EmulateInstructionARM64::AddWithCarry(N, x, y, carry_in, proc_state);
  }
};

class TestAArch64Emulator : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

protected:
};

void TestAArch64Emulator::SetUpTestCase() {
  EmulateInstructionARM64::Initialize();
}

void TestAArch64Emulator::TearDownTestCase() {
  EmulateInstructionARM64::Terminate();
}

TEST_F(TestAArch64Emulator, TestOverflow) {
  EmulateInstructionARM64::ProcState pstate;
  memset(&pstate, 0, sizeof(pstate));
  uint64_t ll_max = std::numeric_limits<int64_t>::max();
  Arch64EmulatorTester emu;
  ASSERT_EQ(emu.AddWithCarry(64, ll_max, 0, 0, pstate), ll_max);
  ASSERT_EQ(pstate.V, 0ULL);
  ASSERT_EQ(pstate.C, 0ULL);
  ASSERT_EQ(emu.AddWithCarry(64, ll_max, 1, 0, pstate), (uint64_t)(ll_max + 1));
  ASSERT_EQ(pstate.V, 1ULL);
  ASSERT_EQ(pstate.C, 0ULL);
  ASSERT_EQ(emu.AddWithCarry(64, ll_max, 0, 1, pstate), (uint64_t)(ll_max + 1));
  ASSERT_EQ(pstate.V, 1ULL);
  ASSERT_EQ(pstate.C, 0ULL);
}
