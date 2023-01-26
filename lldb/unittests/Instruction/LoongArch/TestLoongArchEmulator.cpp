//===-- TestLoongArchEmulator.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/RegisterValue.h"
#include "gtest/gtest.h"

#include "Plugins/Instruction/LoongArch/EmulateInstructionLoongArch.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_loongarch64.h"
#include "Plugins/Process/Utility/lldb-loongarch-register-enums.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

#define GEN_BCOND_TEST(bit, name, rj_val, rd_val_branched, rd_val_continued)   \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##branched) {               \
    testBcondBranch(this, name, true, rj_val, rd_val_branched);                \
  }                                                                            \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##continued) {              \
    testBcondBranch(this, name, false, rj_val, rd_val_continued);              \
  }

#define GEN_BZCOND_TEST(bit, name, rj_val_branched, rj_val_continued)          \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##branched) {               \
    testBZcondBranch(this, name, true, rj_val_branched);                       \
  }                                                                            \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##continued) {              \
    testBZcondBranch(this, name, false, rj_val_continued);                     \
  }

#define GEN_BCZCOND_TEST(bit, name, cj_val_branched, cj_val_continued)         \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##branched) {               \
    testBCZcondBranch(this, name, true, cj_val_branched);                      \
  }                                                                            \
  TEST_F(LoongArch##bit##EmulatorTester, test##name##continued) {              \
    testBCZcondBranch(this, name, false, cj_val_continued);                    \
  }

struct LoongArch64EmulatorTester : public EmulateInstructionLoongArch,
                                   testing::Test {
  RegisterInfoPOSIX_loongarch64::GPR gpr;
  RegisterInfoPOSIX_loongarch64::FPR fpr;

  LoongArch64EmulatorTester(
      std::string triple = "loongarch64-unknown-linux-gnu")
      : EmulateInstructionLoongArch(ArchSpec(triple)) {
    EmulateInstruction::SetReadRegCallback(ReadRegisterCallback);
    EmulateInstruction::SetWriteRegCallback(WriteRegisterCallback);
  }

  static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                   const RegisterInfo *reg_info,
                                   RegisterValue &reg_value) {
    LoongArch64EmulatorTester *tester =
        (LoongArch64EmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg >= gpr_r0_loongarch && reg <= gpr_r31_loongarch)
      reg_value.SetUInt(tester->gpr.gpr[reg], reg_info->byte_size);
    else if (reg == gpr_orig_a0_loongarch)
      reg_value.SetUInt(tester->gpr.orig_a0, reg_info->byte_size);
    else if (reg == gpr_pc_loongarch)
      reg_value.SetUInt(tester->gpr.csr_era, reg_info->byte_size);
    else if (reg == gpr_badv_loongarch)
      reg_value.SetUInt(tester->gpr.csr_badv, reg_info->byte_size);
    else if (reg == fpr_first_loongarch + 32)
      // fcc0
      reg_value.SetUInt(tester->fpr.fcc, reg_info->byte_size);
    return true;
  }

  static bool WriteRegisterCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    const RegisterInfo *reg_info,
                                    const RegisterValue &reg_value) {
    LoongArch64EmulatorTester *tester =
        (LoongArch64EmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg >= gpr_r0_loongarch && reg <= gpr_r31_loongarch)
      tester->gpr.gpr[reg] = reg_value.GetAsUInt64();
    else if (reg == gpr_orig_a0_loongarch)
      tester->gpr.orig_a0 = reg_value.GetAsUInt64();
    else if (reg == gpr_pc_loongarch)
      tester->gpr.csr_era = reg_value.GetAsUInt64();
    else if (reg == gpr_badv_loongarch)
      tester->gpr.csr_badv = reg_value.GetAsUInt64();
    return true;
  }
};

// BEQ BNE BLT BGE BLTU BGEU
static uint32_t EncodeBcondType(uint32_t opcode, uint32_t rj, uint32_t rd,
                                uint32_t offs16) {
  offs16 = offs16 & 0x0000ffff;
  return opcode << 26 | offs16 << 10 | rj << 5 | rd;
}

static uint32_t BEQ(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b010110, rj, rd, uint32_t(offs16));
}

static uint32_t BNE(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b010111, rj, rd, uint32_t(offs16));
}

static uint32_t BLT(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b011000, rj, rd, uint32_t(offs16));
}

static uint32_t BGE(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b011001, rj, rd, uint32_t(offs16));
}

static uint32_t BLTU(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b011010, rj, rd, uint32_t(offs16));
}

static uint32_t BGEU(uint32_t rj, uint32_t rd, int32_t offs16) {
  return EncodeBcondType(0b011011, rj, rd, uint32_t(offs16));
}

// BEQZ BNEZ
static uint32_t EncodeBZcondType(uint32_t opcode, uint32_t rj,
                                 uint32_t offs21) {
  uint32_t offs20_16 = (offs21 & 0x001f0000) >> 16;
  uint32_t offs15_0 = offs21 & 0x0000ffff;
  return opcode << 26 | offs15_0 << 10 | rj << 5 | offs20_16;
}

static uint32_t BEQZ(uint32_t rj, int32_t offs21) {
  return EncodeBZcondType(0b010000, rj, uint32_t(offs21));
}

static uint32_t BNEZ(uint32_t rj, int32_t offs21) {
  return EncodeBZcondType(0b010001, rj, uint32_t(offs21));
}

// BCEQZ BCNEZ
static uint32_t EncodeBCZcondType(uint32_t opcode, uint8_t cj,
                                  uint32_t offs21) {
  uint32_t offs20_16 = (offs21 & 0x001f0000) >> 16;
  uint32_t offs15_0 = offs21 & 0x0000ffff;
  return (opcode >> 2) << 26 | offs15_0 << 10 | (opcode & 0b11) << 8 | cj << 5 |
         offs20_16;
}

static uint32_t BCEQZ(uint8_t cj, int32_t offs21) {
  return EncodeBCZcondType(0b01001000, cj, uint32_t(offs21));
}

static uint32_t BCNEZ(uint8_t cj, int32_t offs21) {
  return EncodeBCZcondType(0b01001001, cj, uint32_t(offs21));
}

using EncoderBcond = uint32_t (*)(uint32_t rj, uint32_t rd, int32_t offs16);
using EncoderBZcond = uint32_t (*)(uint32_t rj, int32_t offs21);
using EncoderBCZcond = uint32_t (*)(uint8_t cj, int32_t offs21);

TEST_F(LoongArch64EmulatorTester, testJIRL) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  WritePC(old_pc);
  // JIRL r1, r12, 0x10
  // | 31       26 | 25                           15 | 9       5 | 4       0 |
  // | 0 1 0 0 1 1 | 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 | 0 1 1 0 0 | 0 0 0 0 1 |
  uint32_t inst = 0b01001100000000000100000110000001;
  uint32_t offs16 = 0x10;
  gpr.gpr[12] = 0x12000400;
  ASSERT_TRUE(TestExecute(inst));
  auto r1 = gpr.gpr[1];
  auto pc = ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(r1, old_pc + 4);
  ASSERT_EQ(pc, gpr.gpr[12] + (offs16 * 4));
}

TEST_F(LoongArch64EmulatorTester, testB) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  WritePC(old_pc);
  // B  0x10010
  // | 31       26 | 25                           10 | 9                 0 |
  // | 0 1 0 1 0 0 | 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 | 0 0 0 0 0 0 0 0 0 1 |
  uint32_t inst = 0b01010000000000000100000000000001;
  uint32_t offs26 = 0x10010;
  ASSERT_TRUE(TestExecute(inst));
  auto pc = ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(pc, old_pc + (offs26 * 4));
}

TEST_F(LoongArch64EmulatorTester, testBL) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  WritePC(old_pc);
  // BL  0x10010
  // | 31       26 | 25                           10 | 9                 0 |
  // | 0 1 0 1 0 1 | 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 | 0 0 0 0 0 0 0 0 0 1 |
  uint32_t inst = 0b01010100000000000100000000000001;
  uint32_t offs26 = 0x10010;
  ASSERT_TRUE(TestExecute(inst));
  auto r1 = gpr.gpr[1];
  auto pc = ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(r1, old_pc + 4);
  ASSERT_EQ(pc, old_pc + (offs26 * 4));
}

static void testBcondBranch(LoongArch64EmulatorTester *tester,
                            EncoderBcond encoder, bool branched,
                            uint64_t rj_val, uint64_t rd_val) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  tester->WritePC(old_pc);
  tester->gpr.gpr[12] = rj_val;
  tester->gpr.gpr[13] = rd_val;
  // b<cmp> r12, r13, (-256)
  uint32_t inst = encoder(12, 13, -256);
  ASSERT_TRUE(tester->TestExecute(inst));
  auto pc = tester->ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(pc, old_pc + (branched ? (-256 * 4) : 4));
}

static void testBZcondBranch(LoongArch64EmulatorTester *tester,
                             EncoderBZcond encoder, bool branched,
                             uint64_t rj_val) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  tester->WritePC(old_pc);
  tester->gpr.gpr[4] = rj_val;
  // b<cmp>z  r4, (-256)
  uint32_t inst = encoder(4, -256);
  ASSERT_TRUE(tester->TestExecute(inst));
  auto pc = tester->ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(pc, old_pc + (branched ? (-256 * 4) : 4));
}

static void testBCZcondBranch(LoongArch64EmulatorTester *tester,
                              EncoderBCZcond encoder, bool branched,
                              uint32_t cj_val) {
  bool success = false;
  addr_t old_pc = 0x12000600;
  tester->WritePC(old_pc);
  tester->fpr.fcc = cj_val;
  // bc<cmp>z fcc0, 256
  uint32_t inst = encoder(0, 256);
  ASSERT_TRUE(tester->TestExecute(inst));
  auto pc = tester->ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(pc, old_pc + (branched ? (256 * 4) : 4));
}

GEN_BCOND_TEST(64, BEQ, 1, 1, 0)
GEN_BCOND_TEST(64, BNE, 1, 0, 1)
GEN_BCOND_TEST(64, BLT, -2, 1, -3)
GEN_BCOND_TEST(64, BGE, -2, -3, 1)
GEN_BCOND_TEST(64, BLTU, -2, -1, 1)
GEN_BCOND_TEST(64, BGEU, -2, 1, -1)
GEN_BZCOND_TEST(64, BEQZ, 0, 1)
GEN_BZCOND_TEST(64, BNEZ, 1, 0)
GEN_BCZCOND_TEST(64, BCEQZ, 0, 1)
GEN_BCZCOND_TEST(64, BCNEZ, 1, 0)
