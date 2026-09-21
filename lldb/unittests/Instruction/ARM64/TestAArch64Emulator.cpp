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
#include "lldb/Core/Opcode.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

struct Arch64EmulatorTester : public EmulateInstructionARM64 {
  RegisterInfoPOSIX_arm64::GPR gpr;
  uint8_t memory[64] = {0};
  uint64_t memory_offset = 0;

  /// The register writes seen since the last Run(), with the context the
  /// emulator attached to each one.
  struct RegisterWrite {
    uint32_t reg = LLDB_INVALID_REGNUM;
    uint64_t value = 0;
    EmulateInstruction::ContextType context_type =
        EmulateInstruction::eContextInvalid;
    /// The register named by an eInfoTypeRegisterPlusOffset payload, if any.
    uint32_t context_reg = LLDB_INVALID_REGNUM;
    int64_t context_offset = 0;
  };
  std::vector<RegisterWrite> writes;

  Arch64EmulatorTester()
      : EmulateInstructionARM64(ArchSpec("arm64-apple-ios")) {
    memset(&gpr, 0, sizeof(gpr));
    EmulateInstruction::SetCallbacks(ReadMemoryCallback, WriteMemoryCallback,
                                     ReadRegisterCallback,
                                     WriteRegisterCallback);
  }

  /// Emulate a single instruction word, discarding any previously recorded
  /// writes. Conditions are always ignored.
  bool Run(uint32_t inst) {
    writes.clear();
    if (!SetInstruction(lldb_private::Opcode(inst, eByteOrderLittle), Address(),
                        nullptr))
      return false;
    return EvaluateInstruction(eEmulateInstructionOptionIgnoreConditions);
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
    RegisterWrite write;
    write.reg = reg;
    write.value = reg_value.GetAsUInt64();
    write.context_type = context.type;
    if (context.GetInfoType() ==
        EmulateInstruction::eInfoTypeRegisterPlusOffset) {
      write.context_reg =
          context.info.RegisterPlusOffset.reg.kinds[eRegisterKindLLDB];
      write.context_offset = context.info.RegisterPlusOffset.signed_offset;
    }
    tester->writes.push_back(write);
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

/// Test that moving the contents from one register to another works.
TEST_F(TestAArch64Emulator, TestMOVRegister) {
  Arch64EmulatorTester emu;
  // lr is x30
  emu.gpr.lr = 0xdeadbeef12345678;

  // mov x2, x30
  ASSERT_TRUE(emu.Run(0xaa1e03e2));

  ASSERT_EQ(1u, emu.writes.size());
  // Check that x2 was written to.
  EXPECT_EQ((uint32_t)gpr_x2_arm64, emu.writes[0].reg);
  // Check that x2 has the value originally on x30.
  EXPECT_EQ(0xdeadbeef12345678ULL, emu.gpr.x[2]);
  // Check that the contents of the write originated from a register.
  EXPECT_EQ(EmulateInstruction::eContextRegisterPlusOffset,
            emu.writes[0].context_type);
  // Check that the original register was lr.
  EXPECT_EQ((uint32_t)gpr_lr_arm64, emu.writes[0].context_reg);
  EXPECT_EQ(0, emu.writes[0].context_offset);
}

TEST_F(TestAArch64Emulator, TestMOVRegister32BitZeroExtends) {
  Arch64EmulatorTester emu;
  emu.gpr.lr = 0xffffffffdeadbeef;

  // mov w2, w30.
  ASSERT_TRUE(emu.Run(0x2a1e03e2));

  ASSERT_EQ(1u, emu.writes.size());
  EXPECT_EQ((uint32_t)gpr_x2_arm64, emu.writes[0].reg);
  EXPECT_EQ(0x00000000deadbeefULL, emu.writes[0].value);
}

TEST_F(TestAArch64Emulator, TestMOVRegisterFromZeroRegister) {
  Arch64EmulatorTester emu;
  // Register 31 is xzr in this encoding, but sp in lldb's numbering.
  // Reading it must yield zero, not the stack pointer.
  emu.gpr.sp = 0x7fff0000;

  // mov x2, xzr
  ASSERT_TRUE(emu.Run(0xaa1f03e2));

  ASSERT_EQ(1u, emu.writes.size());
  EXPECT_EQ((uint32_t)gpr_x2_arm64, emu.writes[0].reg);
  EXPECT_EQ(0u, emu.writes[0].value);
  // There is no source register to name, so the value is an immediate zero.
  EXPECT_EQ(EmulateInstruction::eContextImmediate, emu.writes[0].context_type);
}

TEST_F(TestAArch64Emulator, TestMOVRegisterToZeroRegisterIsDiscarded) {
  Arch64EmulatorTester emu;
  emu.gpr.x[2] = 0xabcd;
  emu.gpr.sp = 0x7fff0000;

  // mov xzr, x2: the destination is xzr, so the write is discarded. sp holds
  // lldb's register 31 and must be left alone.
  ASSERT_TRUE(emu.Run(0xaa0203ff));

  EXPECT_TRUE(emu.writes.empty());
  EXPECT_EQ(0x7fff0000ULL, emu.gpr.sp);
}
