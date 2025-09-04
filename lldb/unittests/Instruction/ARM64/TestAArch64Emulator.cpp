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
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

using namespace lldb;
using namespace lldb_private;

struct Arch64EmulatorTester : public EmulateInstructionARM64 {
  RegisterInfoPOSIX_arm64::GPR gpr;
  uint8_t memory[64] = {0};
  uint64_t memory_offset = 0;

  Arch64EmulatorTester()
      : EmulateInstructionARM64(ArchSpec("arm64-apple-ios")) {
    memset(&gpr, 0, sizeof(gpr));
    EmulateInstruction::SetCallbacks(ReadMemoryCallback, WriteMemoryCallback,
                                     ReadRegisterCallback,
                                     WriteRegisterCallback);
  }

  static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                   const RegisterInfo *reg_info,
                                   RegisterValue &reg_value) {
    auto *tester = static_cast<Arch64EmulatorTester *>(instruction);
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg >= gpr_x0_arm64 && reg <= gpr_x28_arm64) {
      reg_value.SetUInt64(tester->gpr.x[reg - gpr_x0_arm64]);
      return true;
    }
    if (reg >= gpr_w0_arm64 && reg <= gpr_w28_arm64) {
      reg_value.SetUInt32(tester->gpr.x[reg - gpr_w0_arm64]);
      return true;
    }
    switch (reg) {
    case gpr_fp_arm64:
      reg_value.SetUInt64(tester->gpr.fp);
      return true;
    case gpr_lr_arm64:
      reg_value.SetUInt64(tester->gpr.lr);
      return true;
    case gpr_sp_arm64:
      reg_value.SetUInt64(tester->gpr.sp);
      return true;
    case gpr_pc_arm64:
      reg_value.SetUInt64(tester->gpr.pc);
      return true;
    case gpr_cpsr_arm64:
      reg_value.SetUInt32(tester->gpr.cpsr);
      return true;
    default:
      return false;
    }
  }

  static bool WriteRegisterCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    const RegisterInfo *reg_info,
                                    const RegisterValue &reg_value) {
    auto *tester = static_cast<Arch64EmulatorTester *>(instruction);
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg >= gpr_x0_arm64 && reg <= gpr_x28_arm64) {
      tester->gpr.x[reg - gpr_x0_arm64] = reg_value.GetAsUInt64();
      return true;
    }
    if (reg >= gpr_w0_arm64 && reg <= gpr_w28_arm64) {
      tester->gpr.x[reg - gpr_w0_arm64] = reg_value.GetAsUInt32();
      return true;
    }
    switch (reg) {
    case gpr_fp_arm64:
      tester->gpr.fp = reg_value.GetAsUInt64();
      return true;
    case gpr_lr_arm64:
      tester->gpr.lr = reg_value.GetAsUInt64();
      return true;
    case gpr_sp_arm64:
      tester->gpr.sp = reg_value.GetAsUInt64();
      return true;
    case gpr_pc_arm64:
      tester->gpr.pc = reg_value.GetAsUInt64();
      return true;
    case gpr_cpsr_arm64:
      tester->gpr.cpsr = reg_value.GetAsUInt32();
      return true;
    default:
      return false;
    }
  }

  static size_t ReadMemoryCallback(EmulateInstruction *instruction, void *baton,
                                   const Context &context, addr_t addr,
                                   void *dst, size_t length) {
    auto *tester = static_cast<Arch64EmulatorTester *>(instruction);
    assert(addr >= tester->memory_offset);
    assert(addr - tester->memory_offset + length <= sizeof(tester->memory));
    if (addr >= tester->memory_offset &&
        addr - tester->memory_offset + length <= sizeof(tester->memory)) {
      memcpy(dst, tester->memory + (addr - tester->memory_offset), length);
      return length;
    }
    return 0;
  };

  static size_t WriteMemoryCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    addr_t addr, const void *dst,
                                    size_t length) {
    llvm_unreachable("implement when required");
    return 0;
  };

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

TEST_F(TestAArch64Emulator, TestAutoAdvancePC) {
  Arch64EmulatorTester emu;
  emu.memory_offset = 0x123456789abcde00;
  emu.gpr.pc = 0x123456789abcde00;
  emu.gpr.x[8] = 0x123456789abcde20;
  memcpy(emu.memory, "\x08\x01\x40\xb9", 4);        // ldr w8, [x8]
  memcpy(emu.memory + 0x20, "\x11\x22\x33\x44", 4); // 0x44332211
  ASSERT_TRUE(emu.ReadInstruction());
  ASSERT_TRUE(
      emu.EvaluateInstruction(eEmulateInstructionOptionAutoAdvancePC |
                              eEmulateInstructionOptionIgnoreConditions));
  ASSERT_EQ(emu.gpr.pc, (uint64_t)0x123456789abcde04);
  ASSERT_EQ(emu.gpr.x[8], (uint64_t)0x44332211);
}
