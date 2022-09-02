//===-- TestRISCVEmulator.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Instruction/RISCV/EmulateInstructionRISCV.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"
#include "Plugins/Process/Utility/lldb-riscv-register-enums.h"

using namespace lldb;
using namespace lldb_private;

struct RISCVEmulatorTester : public EmulateInstructionRISCV, testing::Test {
  RegisterInfoPOSIX_riscv64::GPR gpr;

  RISCVEmulatorTester()
      : EmulateInstructionRISCV(ArchSpec("riscv64-unknown-linux-gnu")) {
    EmulateInstruction::SetReadRegCallback(ReadRegisterCallback);
    EmulateInstruction::SetWriteRegCallback(WriteRegisterCallback);
  }

  static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                   const RegisterInfo *reg_info,
                                   RegisterValue &reg_value) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg == gpr_x0_riscv)
      reg_value.SetUInt(0, reg_info->byte_size);
    else
      reg_value.SetUInt(tester->gpr.gpr[reg], reg_info->byte_size);
    return true;
  }

  static bool WriteRegisterCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    const RegisterInfo *reg_info,
                                    const RegisterValue &reg_value) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg != gpr_x0_riscv)
      tester->gpr.gpr[reg] = reg_value.GetAsUInt64();
    return true;
  }
};

TEST_F(RISCVEmulatorTester, testJAL) {
  lldb::addr_t old_pc = 0x114514;
  WritePC(old_pc);
  // jal x1, -6*4
  uint32_t inst = 0b11111110100111111111000011101111;
  ASSERT_TRUE(DecodeAndExecute(inst, false));
  auto x1 = gpr.gpr[1];

  bool success = false;
  auto pc = ReadPC(&success);

  ASSERT_TRUE(success);
  ASSERT_EQ(x1, old_pc + 4);
  ASSERT_EQ(pc, old_pc + (-6 * 4));
}

constexpr uint32_t EncodeIType(uint32_t opcode, uint32_t funct3, uint32_t rd,
                               uint32_t rs1, uint32_t imm) {
  return imm << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode;
}

constexpr uint32_t JALR(uint32_t rd, uint32_t rs1, int32_t offset) {
  return EncodeIType(0b1100111, 0, rd, rs1, uint32_t(offset));
}

TEST_F(RISCVEmulatorTester, testJALR) {
  lldb::addr_t old_pc = 0x114514;
  lldb::addr_t old_x2 = 0x1024;
  WritePC(old_pc);
  gpr.gpr[2] = old_x2;
  // jalr x1, x2(-255)
  uint32_t inst = JALR(1, 2, -255);
  ASSERT_TRUE(DecodeAndExecute(inst, false));
  auto x1 = gpr.gpr[1];

  bool success = false;
  auto pc = ReadPC(&success);

  ASSERT_TRUE(success);
  ASSERT_EQ(x1, old_pc + 4);
  // JALR always zeros the bottom bit of the target address.
  ASSERT_EQ(pc, (old_x2 + (-255)) & (~1));
}

constexpr uint32_t EncodeBType(uint32_t opcode, uint32_t funct3, uint32_t rs1,
                               uint32_t rs2, uint32_t imm) {
  uint32_t bimm = (imm & (0b1 << 11)) >> 4 | (imm & (0b11110)) << 7 |
                  (imm & (0b111111 << 5)) << 20 | (imm & (0b1 << 12)) << 19;

  return rs2 << 20 | rs1 << 15 | funct3 << 12 | opcode | bimm;
}

constexpr uint32_t BEQ(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b000, rs1, rs2, uint32_t(offset));
}

constexpr uint32_t BNE(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b001, rs1, rs2, uint32_t(offset));
}

constexpr uint32_t BLT(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b100, rs1, rs2, uint32_t(offset));
}

constexpr uint32_t BGE(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b101, rs1, rs2, uint32_t(offset));
}

constexpr uint32_t BLTU(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b110, rs1, rs2, uint32_t(offset));
}

constexpr uint32_t BGEU(uint32_t rs1, uint32_t rs2, int32_t offset) {
  return EncodeBType(0b1100011, 0b111, rs1, rs2, uint32_t(offset));
}

using EncoderB = uint32_t (*)(uint32_t rs1, uint32_t rs2, int32_t offset);

void testBranch(RISCVEmulatorTester *tester, EncoderB encoder, bool branched,
                uint64_t rs1, uint64_t rs2) {
  // prepare test registers
  lldb::addr_t old_pc = 0x114514;
  tester->WritePC(old_pc);
  tester->gpr.gpr[1] = rs1;
  tester->gpr.gpr[2] = rs2;
  // b<cmp> x1, x2, (-256)
  uint32_t inst = encoder(1, 2, -256);
  ASSERT_TRUE(tester->DecodeAndExecute(inst, false));
  bool success = false;
  auto pc = tester->ReadPC(&success);
  ASSERT_TRUE(success);
  ASSERT_EQ(pc, old_pc + (branched ? (-256) : 0));
}

#define GEN_BRANCH_TEST(name, rs1, rs2_branched, rs2_continued)                \
  TEST_F(RISCVEmulatorTester, test##name##Branched) {                          \
    testBranch(this, name, true, rs1, rs2_branched);                           \
  }                                                                            \
  TEST_F(RISCVEmulatorTester, test##name##Continued) {                         \
    testBranch(this, name, false, rs1, rs2_continued);                         \
  }

void CheckRD(RISCVEmulatorTester *tester, uint64_t rd, uint64_t value) {
  ASSERT_EQ(tester->gpr.gpr[rd], value);
}

using RS1 = uint64_t;
using RS2 = uint64_t;
using PC = uint64_t;
using RDComputer = std::function<uint64_t(RS1, RS2, PC)>;

void TestInst(RISCVEmulatorTester *tester, uint64_t inst, bool has_rs2,
              RDComputer rd_val) {

  lldb::addr_t old_pc = 0x114514;
  tester->WritePC(old_pc);
  auto rd = DecodeRD(inst);
  auto rs1 = DecodeRS1(inst);
  auto rs2 = 0;
  if (rs1)
    tester->gpr.gpr[rs1] = 0x1919;

  if (has_rs2) {
    rs2 = DecodeRS2(inst);
    if (rs2)
      tester->gpr.gpr[rs2] = 0x8181;
  }

  ASSERT_TRUE(tester->DecodeAndExecute(inst, false));
  CheckRD(tester, rd,
          rd_val(tester->gpr.gpr[rs1], rs2 ? tester->gpr.gpr[rs2] : 0, old_pc));
}

// GEN_BRANCH_TEST(opcode, imm1, imm2, imm3):
// It should branch for instruction `opcode imm1, imm2`
// It should do nothing for instruction `opcode imm1, imm3`
GEN_BRANCH_TEST(BEQ, 1, 1, 0)
GEN_BRANCH_TEST(BNE, 1, 0, 1)
GEN_BRANCH_TEST(BLT, -2, 1, -3)
GEN_BRANCH_TEST(BGE, -2, -3, 1)
GEN_BRANCH_TEST(BLTU, -2, -1, 1)
GEN_BRANCH_TEST(BGEU, -2, 1, -1)

struct TestData {
  uint32_t inst;
  std::string name;
  bool has_rs2;
  RDComputer rd_val;
};

TEST_F(RISCVEmulatorTester, TestDecodeAndExcute) {

  std::vector<TestData> tests = {
      {0x00010113, "ADDI", false, [](RS1 rs1, RS2, PC) { return rs1 + 0; }},
      {0x00023517, "AUIPC", false, [](RS1, RS2, PC pc) { return pc + 143360; }},
      {0x0006079b, "ADDIW", false, [](RS1 rs1, RS2, PC) { return rs1 + 0; }},
      {0x00110837, "LUI", false, [](RS1, RS2, PC pc) { return 1114112; }},
      {0x00147513, "ANDI", false, [](RS1 rs1, RS2, PC) { return rs1 & 1; }},
      {0x00153513, "SLTIU", false, [](RS1 rs1, RS2, PC) { return rs1 != 0; }},
      {0x00256513, "ORI", false, [](RS1 rs1, RS2, PC) { return rs1 | 1; }},
      {0x00451a13, "SLLI", false, [](RS1 rs1, RS2, PC) { return rs1 << 4; }},
      {0x00455693, "SRLI", false, [](RS1 rs1, RS2, PC) { return rs1 >> 4; }},
      {0x00a035b3, "SLTU", true, [](RS1 rs1, RS2 rs2, PC) { return rs2 != 0; }},
      {0x00b50633, "ADD", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 + rs2; }},
      {0x40d507b3, "SUB", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 - rs2; }},
  };
  for (auto i : tests) {
    const InstrPattern *pattern = this->Decode(i.inst);
    ASSERT_TRUE(pattern != nullptr);
    std::string name = pattern->name;
    ASSERT_EQ(name, i.name);
    TestInst(this, i.inst, i.has_rs2, i.rd_val);
  }
}
