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
  uint8_t memory[1024] = {0};

  RISCVEmulatorTester()
      : EmulateInstructionRISCV(ArchSpec("riscv64-unknown-linux-gnu")) {
    EmulateInstruction::SetReadRegCallback(ReadRegisterCallback);
    EmulateInstruction::SetWriteRegCallback(WriteRegisterCallback);
    EmulateInstruction::SetReadMemCallback(ReadMemoryCallback);
    EmulateInstruction::SetWriteMemCallback(WriteMemoryCallback);
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

  static size_t ReadMemoryCallback(EmulateInstruction *instruction, void *baton,
                                   const Context &context, lldb::addr_t addr,
                                   void *dst, size_t length) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    assert(addr + length < sizeof(tester->memory));
    memcpy(dst, tester->memory + addr, length);
    return length;
  };

  static size_t WriteMemoryCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    lldb::addr_t addr, const void *dst,
                                    size_t length) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    assert(addr + length < sizeof(tester->memory));
    memcpy(tester->memory + addr, dst, length);
    return length;
  };

  bool DecodeAndExecute(uint32_t inst, bool ignore_cond) {
    return Decode(inst)
        .transform([&](DecodeResult res) { return Execute(res, ignore_cond); })
        .value_or(false);
  }
};

TEST_F(RISCVEmulatorTester, testJAL) {
  lldb::addr_t old_pc = 0x114514;
  WritePC(old_pc);
  // jal x1, -6*4
  uint32_t inst = 0b11111110100111111111000011101111;
  ASSERT_TRUE(DecodeAndExecute(inst, false));
  auto x1 = gpr.gpr[1];
  auto pc = ReadPC();
  ASSERT_TRUE(pc.has_value());
  ASSERT_EQ(x1, old_pc + 4);
  ASSERT_EQ(*pc, old_pc + (-6 * 4));
}

constexpr uint32_t EncodeIType(uint32_t opcode, uint32_t funct3, uint32_t rd,
                               uint32_t rs1, uint32_t imm) {
  return imm << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode;
}

constexpr uint32_t EncodeJALR(uint32_t rd, uint32_t rs1, int32_t offset) {
  return EncodeIType(0b1100111, 0, rd, rs1, uint32_t(offset));
}

TEST_F(RISCVEmulatorTester, testJALR) {
  lldb::addr_t old_pc = 0x114514;
  lldb::addr_t old_x2 = 0x1024;
  WritePC(old_pc);
  gpr.gpr[2] = old_x2;
  // jalr x1, x2(-255)
  uint32_t inst = EncodeJALR(1, 2, -255);
  ASSERT_TRUE(DecodeAndExecute(inst, false));
  auto x1 = gpr.gpr[1];
  auto pc = ReadPC();
  ASSERT_TRUE(pc.has_value());
  ASSERT_EQ(x1, old_pc + 4);
  // JALR always zeros the bottom bit of the target address.
  ASSERT_EQ(*pc, (old_x2 + (-255)) & (~1));
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
  auto pc = tester->ReadPC();
  ASSERT_TRUE(pc.has_value());
  ASSERT_EQ(*pc, old_pc + (branched ? (-256) : 0));
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

template <typename T>
void CheckMem(RISCVEmulatorTester *tester, uint64_t addr, uint64_t value) {
  auto mem = tester->ReadMem<T>(addr);
  ASSERT_TRUE(mem.has_value());
  ASSERT_EQ(*mem, value);
}

using RS1 = uint64_t;
using RS2 = uint64_t;
using PC = uint64_t;
using RDComputer = std::function<uint64_t(RS1, RS2, PC)>;

void TestInst(RISCVEmulatorTester *tester, DecodeResult inst, bool has_rs2,
              RDComputer rd_val) {

  lldb::addr_t old_pc = 0x114514;
  tester->WritePC(old_pc);
  uint32_t rd = DecodeRD(inst.inst);
  uint32_t rs1 = DecodeRS1(inst.inst);
  uint32_t rs2 = 0;

  uint64_t rs1_val = 0x19;
  uint64_t rs2_val = 0x81;

  if (rs1)
    tester->gpr.gpr[rs1] = rs1_val;

  if (has_rs2) {
    rs2 = DecodeRS2(inst.inst);
    if (rs2) {
      if (rs1 == rs2)
        rs2_val = rs1_val;
      tester->gpr.gpr[rs2] = rs2_val;
    }
  }

  ASSERT_TRUE(tester->Execute(inst, false));
  CheckRD(tester, rd, rd_val(rs1_val, rs2 ? rs2_val : 0, old_pc));
}

template <typename T>
void TestAtomic(RISCVEmulatorTester *tester, uint64_t inst, T rs1_val,
                T rs2_val, T rd_expected, T mem_expected) {
  // Atomic inst must have rs1 and rs2

  uint32_t rd = DecodeRD(inst);
  uint32_t rs1 = DecodeRS1(inst);
  uint32_t rs2 = DecodeRS2(inst);

  // addr was stored in rs1
  uint64_t atomic_addr = 0x100;

  tester->gpr.gpr[rs1] = atomic_addr;
  tester->gpr.gpr[rs2] = rs2_val;

  // Write and check rs1_val in atomic_addr
  ASSERT_TRUE(tester->WriteMem<T>(atomic_addr, rs1_val));
  CheckMem<T>(tester, atomic_addr, rs1_val);

  ASSERT_TRUE(tester->DecodeAndExecute(inst, false));
  CheckRD(tester, rd, rd_expected);
  CheckMem<T>(tester, atomic_addr, mem_expected);
}

TEST_F(RISCVEmulatorTester, TestAtomicSequence) {
  this->WritePC(0x0);
  *(uint32_t *)this->memory = 0x100427af;        // lr.w	a5,(s0)
  *(uint32_t *)(this->memory + 4) = 0x00079663;  // bnez	a5,12
  *(uint32_t *)(this->memory + 8) = 0x1ce426af;  // sc.w.aq	a3,a4,(s0)
  *(uint32_t *)(this->memory + 12) = 0xfe069ae3; // bnez	a3,-12
  ASSERT_TRUE(this->DecodeAndExecute(*(uint32_t *)this->memory, false));
  ASSERT_EQ(this->gpr.gpr[0], uint64_t(16));
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
      // RV32I & RV64I Tests
      {0x00010113, "ADDI", false, [](RS1 rs1, RS2, PC) { return rs1 + 0; }},
      {0x00023517, "AUIPC", false, [](RS1, RS2, PC pc) { return pc + 143360; }},
      {0x0006079b, "ADDIW", false, [](RS1 rs1, RS2, PC) { return rs1 + 0; }},
      {0x00110837, "LUI", false, [](RS1, RS2, PC pc) { return 1114112; }},
      {0x00147513, "ANDI", false, [](RS1 rs1, RS2, PC) { return rs1 & 1; }},
      {0x00153513, "SLTIU", false, [](RS1 rs1, RS2, PC) { return 0; }},
      {0x00256513, "ORI", false, [](RS1 rs1, RS2, PC) { return rs1 | 2; }},
      {0x00451a13, "SLLI", false, [](RS1 rs1, RS2, PC) { return rs1 << 4; }},
      {0x00455693, "SRLI", false, [](RS1 rs1, RS2, PC) { return rs1 >> 4; }},
      {0x00a035b3, "SLTU", true, [](RS1 rs1, RS2 rs2, PC) { return rs2 != 0; }},
      {0x00b50633, "ADD", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 + rs2; }},
      {0x40d507b3, "SUB", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 - rs2; }},

      // RV32M & RV64M Tests
      {0x02f787b3, "MUL", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 * rs2; }},
      {0x2F797B3, "MULH", true, [](RS1 rs1, RS2 rs2, PC) { return 0; }},
      {0x2F7A7B3, "MULHSU", true, [](RS1 rs1, RS2 rs2, PC) { return 0; }},
      {0x2F7B7B3, "MULHU", true, [](RS1 rs1, RS2 rs2, PC) { return 0; }},
      {0x02f747b3, "DIV", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 / rs2; }},
      {0x02f757b3, "DIVU", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 / rs2; }},
      {0x02f767b3, "REM", true, [](RS1 rs1, RS2 rs2, PC) { return rs1 % rs2; }},
      {0x02f777b3, "REMU", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 % rs2; }},
      {0x02f787bb, "MULW", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 * rs2; }},
      {0x02f747bb, "DIVW", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 / rs2; }},
      {0x02f757bb, "DIVUW", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 / rs2; }},
      {0x02f767bb, "REMW", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 % rs2; }},
      {0x02f777bb, "REMUW", true,
       [](RS1 rs1, RS2 rs2, PC) { return rs1 % rs2; }},
  };
  for (auto i : tests) {
    auto decode = this->Decode(i.inst);
    ASSERT_TRUE(decode.has_value());
    std::string name = decode->pattern.name;
    ASSERT_EQ(name, i.name);
    TestInst(this, *decode, i.has_rs2, i.rd_val);
  }
}

TEST_F(RISCVEmulatorTester, TestAMOSWAP) {
  TestAtomic<uint32_t>(this, 0x8F7282F, 0x1, 0x2, 0x1, 0x2);
  TestAtomic<uint64_t>(this, 0x8F7382F, 0x1, 0x2, 0x1, 0x2);
}

TEST_F(RISCVEmulatorTester, TestAMOADD) {
  TestAtomic<uint32_t>(this, 0xF7282F, 0x1, 0x2, 0x1, 0x3);
  TestAtomic<uint64_t>(this, 0xF7382F, 0x1, 0x2, 0x1, 0x3);
}

TEST_F(RISCVEmulatorTester, TestAMOXOR) {
  TestAtomic<uint32_t>(this, 0x20F7282F, 0x1, 0x2, 0x1, 0x3);
  TestAtomic<uint32_t>(this, 0x20F7382F, 0x1, 0x2, 0x1, 0x3);
}

TEST_F(RISCVEmulatorTester, TestAMOAND) {
  TestAtomic<uint32_t>(this, 0x60F7282F, 0x1, 0x2, 0x1, 0x0);
  TestAtomic<uint64_t>(this, 0x60F7382F, 0x1, 0x2, 0x1, 0x0);
}

TEST_F(RISCVEmulatorTester, TestAMOOR) {
  TestAtomic<uint32_t>(this, 0x40F7282F, 0x1, 0x2, 0x1, 0x3);
  TestAtomic<uint32_t>(this, 0x40F7382F, 0x1, 0x2, 0x1, 0x3);
}

TEST_F(RISCVEmulatorTester, TestAMOMIN) {
  TestAtomic<uint32_t>(this, 0x80F7282F, 0x1, 0x2, 0x1, 0x1);
  TestAtomic<uint64_t>(this, 0x80F7382F, 0x1, 0x2, 0x1, 0x1);
}

TEST_F(RISCVEmulatorTester, TestAMOMAX) {
  TestAtomic<uint32_t>(this, 0xA0F7282F, 0x1, 0x2, 0x1, 0x2);
  TestAtomic<uint64_t>(this, 0xA0F7382F, 0x1, 0x2, 0x1, 0x2);
}

TEST_F(RISCVEmulatorTester, TestAMOMINU) {
  TestAtomic<uint32_t>(this, 0xC0F7282F, 0x1, 0x2, 0x1, 0x1);
  TestAtomic<uint64_t>(this, 0xC0F7382F, 0x1, 0x2, 0x1, 0x1);
}

TEST_F(RISCVEmulatorTester, TestAMOMAXU) {
  TestAtomic<uint32_t>(this, 0xE0F7282F, 0x1, 0x2, 0x1, 0x2);
  TestAtomic<uint64_t>(this, 0xE0F7382F, 0x1, 0x2, 0x1, 0x2);
}
