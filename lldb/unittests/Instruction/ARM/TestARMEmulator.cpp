//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Utility/ARM_DWARF_Registers.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/RegisterValue.h"

#include <map>

using namespace lldb;
using namespace lldb_private;

struct ARMEmulatorTester : public EmulateInstructionARM {
  uint32_t gpr[17] = {0}; // r0-r15 + cpsr
  std::map<addr_t, uint32_t> memory;

  ARMEmulatorTester() : EmulateInstructionARM(ArchSpec("thumbv7-apple-ios")) {
    EmulateInstruction::SetReadRegCallback(ReadRegisterCallback);
    EmulateInstruction::SetWriteRegCallback(WriteRegisterCallback);
    EmulateInstruction::SetReadMemCallback(ReadMemoryCallback);
    EmulateInstruction::SetWriteMemCallback(WriteMemoryCallback);
  }

  static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                   const RegisterInfo *reg_info,
                                   RegisterValue &reg_value) {
    auto *tester = static_cast<ARMEmulatorTester *>(instruction);
    uint32_t reg = reg_info->kinds[eRegisterKindDWARF];
    if (reg <= dwarf_cpsr) {
      reg_value.SetUInt32(tester->gpr[reg - dwarf_r0]);
      return true;
    }
    return false;
  }

  static bool WriteRegisterCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    const RegisterInfo *reg_info,
                                    const RegisterValue &reg_value) {
    auto *tester = static_cast<ARMEmulatorTester *>(instruction);
    uint32_t reg = reg_info->kinds[eRegisterKindDWARF];
    if (reg <= dwarf_cpsr) {
      tester->gpr[reg - dwarf_r0] = reg_value.GetAsUInt32();
      return true;
    }
    return false;
  }

  static size_t ReadMemoryCallback(EmulateInstruction *instruction, void *baton,
                                   const Context &context, addr_t addr,
                                   void *dst, size_t length) {
    auto *tester = static_cast<ARMEmulatorTester *>(instruction);
    // Read word-by-word from the memory map.
    for (size_t i = 0; i < length; i++) {
      addr_t word_addr = (addr + i) & ~3ULL;
      auto it = tester->memory.find(word_addr);
      if (it == tester->memory.end())
        return 0;
      uint32_t byte_offset = (addr + i) & 3;
      static_cast<uint8_t *>(dst)[i] = (it->second >> (byte_offset * 8)) & 0xff;
    }
    return length;
  }

  static size_t WriteMemoryCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    addr_t addr, const void *dst,
                                    size_t length) {
    auto *tester = static_cast<ARMEmulatorTester *>(instruction);
    if (length == 4 && (addr & 3) == 0) {
      uint32_t value;
      memcpy(&value, dst, 4);
      tester->memory[addr] = value;
      return 4;
    }
    return 0;
  }

  bool TestEmulateSTRThumb(const uint32_t opcode, const ARMEncoding encoding) {
    return EmulateSTRThumb(opcode, encoding);
  }

  void SetupThumbMode(uint16_t opcode16) {
    m_opcode_mode = eModeThumb;
    m_opcode_cpsr = 0;
    m_opcode.SetOpcode16(opcode16, lldb::eByteOrderLittle);
    m_ignore_conditions = true;
  }
};

class TestARMEmulator : public testing::Test {
public:
  static void SetUpTestCase() { EmulateInstructionARM::Initialize(); }
  static void TearDownTestCase() { EmulateInstructionARM::Terminate(); }
};

// Test that STR (Thumb T1 encoding) uses add=true, computing address as
// Rn + imm5*4, not Rn - imm5*4.
//
// STR r0, [r1, #4]  =>  Thumb T1 encoding: 0110 0 imm5 Rn Rt
//   imm5 = 1 (offset = 1*4 = 4), Rn = r1, Rt = r0
//   opcode = 0110 0 00001 001 000 = 0x6048
TEST_F(TestARMEmulator, TestSTRT1AddsOffset) {
  ARMEmulatorTester emu;

  // Set up: r0 = 0xDEADBEEF (value to store), r1 = 0x1000 (base address)
  emu.gpr[dwarf_r0] = 0xDEADBEEF;
  emu.gpr[dwarf_r1] = 0x1000;
  // Set up internal emulator state for Thumb mode.
  emu.SetupThumbMode(0x6048);

  // STR r0, [r1, #4] — T1 encoding
  // With add=true, should store to 0x1000 + 4 = 0x1004
  // With the old bug (add=false), it would store to 0x1000 - 4 = 0x0FFC
  const uint32_t opcode = 0x6048;
  ASSERT_TRUE(
      emu.TestEmulateSTRThumb(opcode, EmulateInstructionARM::eEncodingT1));

  // Verify the value was stored at base + offset (0x1004), not base - offset
  auto it = emu.memory.find(0x1004);
  ASSERT_NE(it, emu.memory.end());
  ASSERT_EQ(it->second, (uint32_t)0xDEADBEEF);

  // Verify nothing was written to the wrong address (0x0FFC)
  ASSERT_EQ(emu.memory.find(0x0FFC), emu.memory.end());
}
