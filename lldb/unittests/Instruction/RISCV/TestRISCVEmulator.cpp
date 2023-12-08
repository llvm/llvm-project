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

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

struct RISCVEmulatorTester : public EmulateInstructionRISCV, testing::Test {
  RegisterInfoPOSIX_riscv64::GPR gpr;
  RegisterInfoPOSIX_riscv64::FPR fpr;
  uint8_t memory[1024] = {0};

  RISCVEmulatorTester(std::string triple = "riscv64-unknown-linux-gnu")
      : EmulateInstructionRISCV(ArchSpec(triple)) {
    EmulateInstruction::SetReadRegCallback(ReadRegisterCallback);
    EmulateInstruction::SetWriteRegCallback(WriteRegisterCallback);
    EmulateInstruction::SetReadMemCallback(ReadMemoryCallback);
    EmulateInstruction::SetWriteMemCallback(WriteMemoryCallback);
    ClearAll();
  }

  static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                   const RegisterInfo *reg_info,
                                   RegisterValue &reg_value) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg == gpr_x0_riscv)
      reg_value.SetUInt(0, reg_info->byte_size);
    if (reg >= gpr_pc_riscv && reg <= gpr_x31_riscv)
      reg_value.SetUInt(tester->gpr.gpr[reg], reg_info->byte_size);
    if (reg >= fpr_f0_riscv && reg <= fpr_f31_riscv)
      reg_value.SetUInt(tester->fpr.fpr[reg - fpr_f0_riscv],
                        reg_info->byte_size);
    if (reg == fpr_fcsr_riscv)
      reg_value.SetUInt(tester->fpr.fcsr, reg_info->byte_size);
    return true;
  }

  static bool WriteRegisterCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    const RegisterInfo *reg_info,
                                    const RegisterValue &reg_value) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    if (reg >= gpr_pc_riscv && reg <= gpr_x31_riscv)
      tester->gpr.gpr[reg] = reg_value.GetAsUInt64();
    if (reg >= fpr_f0_riscv && reg <= fpr_f31_riscv)
      tester->fpr.fpr[reg - fpr_f0_riscv] = reg_value.GetAsUInt64();
    if (reg == fpr_fcsr_riscv)
      tester->fpr.fcsr = reg_value.GetAsUInt32();
    return true;
  }

  static size_t ReadMemoryCallback(EmulateInstruction *instruction, void *baton,
                                   const Context &context, addr_t addr,
                                   void *dst, size_t length) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    assert(addr + length < sizeof(tester->memory));
    memcpy(dst, tester->memory + addr, length);
    return length;
  };

  static size_t WriteMemoryCallback(EmulateInstruction *instruction,
                                    void *baton, const Context &context,
                                    addr_t addr, const void *dst,
                                    size_t length) {
    RISCVEmulatorTester *tester = (RISCVEmulatorTester *)instruction;
    assert(addr + length < sizeof(tester->memory));
    memcpy(tester->memory + addr, dst, length);
    return length;
  };

  bool DecodeAndExecute(uint32_t inst, bool ignore_cond) {
    return llvm::transformOptional(
               Decode(inst),
               [&](DecodeResult res) { return Execute(res, ignore_cond); })
        .value_or(false);
  }

  void ClearAll() {
    memset(&gpr, 0, sizeof(gpr));
    memset(&fpr, 0, sizeof(fpr));
    memset(memory, 0, sizeof(memory));
  }
};

TEST_F(RISCVEmulatorTester, testJAL) {
  addr_t old_pc = 0x114514;
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
  addr_t old_pc = 0x114514;
  addr_t old_x2 = 0x1024;
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

static void testBranch(RISCVEmulatorTester *tester, EncoderB encoder,
                       bool branched, uint64_t rs1, uint64_t rs2) {
  // prepare test registers
  addr_t old_pc = 0x114514;
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

static void CheckRD(RISCVEmulatorTester *tester, uint64_t rd, uint64_t value) {
  ASSERT_EQ(tester->gpr.gpr[rd], value);
}

template <typename T>
static void CheckMem(RISCVEmulatorTester *tester, uint64_t addr,
                     uint64_t value) {
  auto mem = tester->ReadMem<T>(addr);
  ASSERT_TRUE(mem.has_value());
  ASSERT_EQ(*mem, value);
}

using RS1 = uint64_t;
using RS2 = uint64_t;
using PC = uint64_t;
using RDComputer = std::function<uint64_t(RS1, RS2, PC)>;

static void TestInst(RISCVEmulatorTester *tester, DecodeResult inst,
                     bool has_rs2, RDComputer rd_val) {

  addr_t old_pc = 0x114514;
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
static void TestAtomic(RISCVEmulatorTester *tester, uint64_t inst, T rs1_val,
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

struct TestDecode {
  uint32_t inst;
  RISCVInst inst_type;
};

TEST_F(RISCVEmulatorTester, TestCDecode) {
  std::vector<TestDecode> tests = {
      {0x0000, INVALID{0x0000}},
      {0x0010, RESERVED{0x0010}},
      // ADDI4SPN here, decode as ADDI
      {0x0024, ADDI{Rd{9}, Rs{2}, 8}},
      {0x2084, FLD{Rd{9}, Rs{9}, 0}},
      {0x4488, LW{Rd{10}, Rs{9}, 8}},
      {0x6488, LD{Rd{10}, Rs{9}, 8}},
      {0xA084, FSD{Rs{9}, Rs{9}, 0}},
      {0xC488, SW{Rs{9}, Rs{10}, 8}},
      {0xE488, SD{Rs{9}, Rs{10}, 8}},
      {0x1001, NOP{0x1001}},
      {0x1085, ADDI{Rd{1}, Rs{1}, uint32_t(-31)}},
      {0x2081, ADDIW{Rd{1}, Rs{1}, 0}},
      // ADDI16SP here, decode as ADDI
      {0x7101, ADDI{Rd{2}, Rs{2}, uint32_t(-512)}},
      {0x4081, ADDI{Rd{1}, Rs{0}, 0}},
      {0x7081, LUI{Rd{1}, uint32_t(-131072)}},
      {0x8085, SRLI{Rd{9}, Rs{9}, 1}},
      {0x8485, SRAI{Rd{9}, Rs{9}, 1}},
      {0x8881, ANDI{Rd{9}, Rs{9}, 0}},
      {0x8C85, SUB{Rd{9}, Rs{9}, Rs{9}}},
      {0x8CA5, XOR{Rd{9}, Rs{9}, Rs{9}}},
      {0x8CC5, OR{Rd{9}, Rs{9}, Rs{9}}},
      {0x8CE5, AND{Rd{9}, Rs{9}, Rs{9}}},
      {0x9C85, SUBW{Rd{9}, Rs{9}, Rs{9}}},
      {0x9CA5, ADDW{Rd{9}, Rs{9}, Rs{9}}},
      // C.J here, decoded as JAL
      {0xA001, JAL{Rd{0}, 0}},
      {0xC081, B{Rs{9}, Rs{0}, 0, 0b000}},
      {0xE081, B{Rs{9}, Rs{0}, 0, 0b001}},
      {0x1082, SLLI{Rd{1}, Rs{1}, 32}},
      {0x1002, HINT{0x1002}},
      // SLLI64 here, decoded as HINT if not in RV128
      {0x0082, HINT{0x0082}},
      // FLDSP here, decoded as FLD
      {0x2082, FLD{Rd{1}, Rs{2}, 0}},
      // LWSP here, decoded as LW
      {0x4082, LW{Rd{1}, Rs{2}, 0}},
      // LDSP here, decoded as LD
      {0x6082, LD{Rd{1}, Rs{2}, 0}},
      // C.JR here, decoded as JALR
      {0x8082, JALR{Rd{0}, Rs{1}, 0}},
      // C.MV here, decoded as ADD
      {0x8086, ADD{Rd{1}, Rs{0}, Rs{1}}},
      {0x9002, EBREAK{0x9002}},
      {0x9082, JALR{Rd{1}, Rs{1}, 0}},
      {0x9086, ADD{Rd{1}, Rs{1}, Rs{1}}},
      // C.FSDSP here, decoded as FSD
      {0xA006, FSD{Rs{2}, Rs{1}, 0}},
      // C.SWSP here, decoded as SW
      {0xC006, SW{Rs{2}, Rs{1}, 0}},
      // C.SDSP here, decoded as SD
      {0xE006, SD{Rs{2}, Rs{1}, 0}},
  };

  for (auto i : tests) {
    auto decode = this->Decode(i.inst);
    ASSERT_TRUE(decode.has_value());
    ASSERT_EQ(decode->decoded, i.inst_type);
  }
}

class RISCVEmulatorTester32 : public RISCVEmulatorTester {
public:
  RISCVEmulatorTester32() : RISCVEmulatorTester("riscv32-unknown-linux-gnu") {}
};

TEST_F(RISCVEmulatorTester32, TestCDecodeRV32) {
  std::vector<TestDecode> tests = {
      {0x6002, FLW{Rd{0}, Rs{2}, 0}},
      {0xE006, FSW{Rs{2}, Rs{1}, 0}},
      {0x6000, FLW{Rd{8}, Rs{8}, 0}},
      {0xE000, FSW{Rs{8}, Rs{8}, 0}},

      {0x2084, FLD{Rd{9}, Rs{9}, 0}},
      {0xA084, FSD{Rs{9}, Rs{9}, 0}},
      {0x2082, FLD{Rd{1}, Rs{2}, 0}},
      {0xA006, FSD{Rs{2}, Rs{1}, 0}},
  };

  for (auto i : tests) {
    auto decode = this->Decode(i.inst);
    ASSERT_TRUE(decode.has_value());
    ASSERT_EQ(decode->decoded, i.inst_type);
  }
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

template <typename T> struct F_D_CalInst {
  uint32_t inst;
  std::string name;
  T rs1_val;
  T rs2_val;
  T rd_val;
};

using FloatCalInst = F_D_CalInst<float>;
using DoubleCalInst = F_D_CalInst<double>;

template <typename T>
static void TestF_D_CalInst(RISCVEmulatorTester *tester, DecodeResult inst,
                            T rs1_val, T rs2_val, T rd_exp) {
  std::vector<std::string> CMPs = {"FEQ_S", "FLT_S", "FLE_S",
                                   "FEQ_D", "FLT_D", "FLE_D"};
  std::vector<std::string> FMAs = {"FMADD_S",  "FMSUB_S", "FNMSUB_S",
                                   "FNMADD_S", "FMADD_D", "FMSUB_D",
                                   "FNMSUB_D", "FNMADD_D"};

  uint32_t rd = DecodeRD(inst.inst);
  uint32_t rs1 = DecodeRS1(inst.inst);
  uint32_t rs2 = DecodeRS2(inst.inst);

  APFloat ap_rs1_val(rs1_val);
  APFloat ap_rs2_val(rs2_val);
  APFloat ap_rs3_val(0.0f);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "T should be float or double");
  if constexpr (std::is_same_v<T, float>)
    ap_rs3_val = APFloat(0.5f);
  if constexpr (std::is_same_v<T, double>)
    ap_rs3_val = APFloat(0.5);

  if (rs1)
    tester->fpr.fpr[rs1] = ap_rs1_val.bitcastToAPInt().getZExtValue();
  if (rs2)
    tester->fpr.fpr[rs2] = ap_rs2_val.bitcastToAPInt().getZExtValue();
  for (auto i : FMAs) {
    if (inst.pattern.name == i) {
      uint32_t rs3 = DecodeRS3(inst.inst);
      tester->fpr.fpr[rs3] = ap_rs3_val.bitcastToAPInt().getZExtValue();
    }
  }
  ASSERT_TRUE(tester->Execute(inst, false));
  for (auto i : CMPs) {
    if (inst.pattern.name == i) {
      ASSERT_EQ(tester->gpr.gpr[rd], rd_exp);
      return;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    APInt apInt(32, tester->fpr.fpr[rd]);
    APFloat rd_val(apInt.bitsToFloat());
    ASSERT_EQ(rd_val.convertToFloat(), rd_exp);
  }
  if constexpr (std::is_same_v<T, double>) {
    APInt apInt(64, tester->fpr.fpr[rd]);
    APFloat rd_val(apInt.bitsToDouble());
    ASSERT_EQ(rd_val.convertToDouble(), rd_exp);
  }
}

TEST_F(RISCVEmulatorTester, TestFloatInst) {
  std::vector<FloatCalInst> tests = {
      {0x21F253, "FADD_S", 0.5f, 0.5f, 1.0f},
      {0x821F253, "FSUB_S", 1.0f, 0.5f, 0.5f},
      {0x1021F253, "FMUL_S", 0.5f, 0.5f, 0.25f},
      {0x1821F253, "FDIV_S", 0.1f, 0.1f, 1.0f},
      {0x20218253, "FSGNJ_S", 0.5f, 0.2f, 0.5f},
      {0x20219253, "FSGNJN_S", 0.5f, -1.0f, 0.5f},
      {0x2021A253, "FSGNJX_S", -0.5f, -0.5f, 0.5f},
      {0x2021A253, "FSGNJX_S", -0.5f, 0.5f, -0.5f},
      {0x28218253, "FMIN_S", -0.5f, 0.5f, -0.5f},
      {0x28218253, "FMIN_S", -0.5f, -0.6f, -0.6f},
      {0x28218253, "FMIN_S", 0.5f, 0.6f, 0.5f},
      {0x28219253, "FMAX_S", -0.5f, -0.6f, -0.5f},
      {0x28219253, "FMAX_S", 0.5f, 0.6f, 0.6f},
      {0x28219253, "FMAX_S", 0.5f, -0.6f, 0.5f},
      {0xA021A253, "FEQ_S", 0.5f, 0.5f, 1},
      {0xA021A253, "FEQ_S", 0.5f, -0.5f, 0},
      {0xA021A253, "FEQ_S", -0.5f, 0.5f, 0},
      {0xA021A253, "FEQ_S", 0.4f, 0.5f, 0},
      {0xA0219253, "FLT_S", 0.4f, 0.5f, 1},
      {0xA0219253, "FLT_S", 0.5f, 0.5f, 0},
      {0xA0218253, "FLE_S", 0.4f, 0.5f, 1},
      {0xA0218253, "FLE_S", 0.5f, 0.5f, 1},
      {0x4021F243, "FMADD_S", 0.5f, 0.5f, 0.75f},
      {0x4021F247, "FMSUB_S", 0.5f, 0.5f, -0.25f},
      {0x4021F24B, "FNMSUB_S", 0.5f, 0.5f, 0.25f},
      {0x4021F24F, "FNMADD_S", 0.5f, 0.5f, -0.75f},
  };
  for (auto i : tests) {
    auto decode = this->Decode(i.inst);
    ASSERT_TRUE(decode.has_value());
    std::string name = decode->pattern.name;
    ASSERT_EQ(name, i.name);
    TestF_D_CalInst(this, *decode, i.rs1_val, i.rs2_val, i.rd_val);
  }
}

TEST_F(RISCVEmulatorTester, TestDoubleInst) {
  std::vector<DoubleCalInst> tests = {
      {0x221F253, "FADD_D", 0.5, 0.5, 1.0},
      {0xA21F253, "FSUB_D", 1.0, 0.5, 0.5},
      {0x1221F253, "FMUL_D", 0.5, 0.5, 0.25},
      {0x1A21F253, "FDIV_D", 0.1, 0.1, 1.0},
      {0x22218253, "FSGNJ_D", 0.5, 0.2, 0.5},
      {0x22219253, "FSGNJN_D", 0.5, -1.0, 0.5},
      {0x2221A253, "FSGNJX_D", -0.5, -0.5, 0.5},
      {0x2221A253, "FSGNJX_D", -0.5, 0.5, -0.5},
      {0x2A218253, "FMIN_D", -0.5, 0.5, -0.5},
      {0x2A218253, "FMIN_D", -0.5, -0.6, -0.6},
      {0x2A218253, "FMIN_D", 0.5, 0.6, 0.5},
      {0x2A219253, "FMAX_D", -0.5, -0.6, -0.5},
      {0x2A219253, "FMAX_D", 0.5, 0.6, 0.6},
      {0x2A219253, "FMAX_D", 0.5, -0.6, 0.5},
      {0xA221A253, "FEQ_D", 0.5, 0.5, 1},
      {0xA221A253, "FEQ_D", 0.5, -0.5, 0},
      {0xA221A253, "FEQ_D", -0.5, 0.5, 0},
      {0xA221A253, "FEQ_D", 0.4, 0.5, 0},
      {0xA2219253, "FLT_D", 0.4, 0.5, 1},
      {0xA2219253, "FLT_D", 0.5, 0.5, 0},
      {0xA2218253, "FLE_D", 0.4, 0.5, 1},
      {0xA2218253, "FLE_D", 0.5, 0.5, 1},
      {0x4221F243, "FMADD_D", 0.5, 0.5, 0.75},
      {0x4221F247, "FMSUB_D", 0.5, 0.5, -0.25},
      {0x4221F24B, "FNMSUB_D", 0.5, 0.5, 0.25},
      {0x4221F24F, "FNMADD_D", 0.5, 0.5, -0.75},
  };
  for (auto i : tests) {
    auto decode = this->Decode(i.inst);
    ASSERT_TRUE(decode.has_value());
    std::string name = decode->pattern.name;
    ASSERT_EQ(name, i.name);
    TestF_D_CalInst(this, *decode, i.rs1_val, i.rs2_val, i.rd_val);
  }
}

template <typename T>
static void TestInverse(RISCVEmulatorTester *tester, uint32_t f_reg,
                        uint32_t x_reg, DecodeResult f2i, DecodeResult i2f,
                        APFloat apf_val) {
  uint64_t exp_x;
  if constexpr (std::is_same_v<T, float>)
    exp_x = uint64_t(apf_val.convertToFloat());
  if constexpr (std::is_same_v<T, double>)
    exp_x = uint64_t(apf_val.convertToDouble());
  T exp_f = T(exp_x);

  // convert float/double to int.
  tester->fpr.fpr[f_reg] = apf_val.bitcastToAPInt().getZExtValue();
  ASSERT_TRUE(tester->Execute(f2i, false));
  ASSERT_EQ(tester->gpr.gpr[x_reg], exp_x);

  // then convert int to float/double back.
  ASSERT_TRUE(tester->Execute(i2f, false));
  ASSERT_EQ(tester->fpr.fpr[f_reg],
            APFloat(exp_f).bitcastToAPInt().getZExtValue());
}

struct FCVTInst {
  uint32_t f2i;
  uint32_t i2f;
  APFloat data;
  bool isDouble;
};

TEST_F(RISCVEmulatorTester, TestFCVT) {
  std::vector<FCVTInst> tests{
      // FCVT_W_S and FCVT_S_W
      {0xC000F0D3, 0xD000F0D3, APFloat(12.0f), false},
      // FCVT_WU_S and FCVT_S_WU
      {0xC010F0D3, 0xD010F0D3, APFloat(12.0f), false},
      // FCVT_L_S and FCVT_S_L
      {0xC020F0D3, 0xD020F0D3, APFloat(12.0f), false},
      // FCVT_LU_S and FCVT_S_LU
      {0xC030F0D3, 0xD030F0D3, APFloat(12.0f), false},
      // FCVT_W_D and FCVT_D_W
      {0xC200F0D3, 0xD200F0D3, APFloat(12.0), true},
      // FCVT_WU_D and FCVT_D_WU
      {0xC210F0D3, 0xD210F0D3, APFloat(12.0), true},
      // FCVT_L_D and FCVT_D_L
      {0xC220F0D3, 0xD220F0D3, APFloat(12.0), true},
      // FCVT_LU_D and FCVT_D_LU
      {0xC230F0D3, 0xD230F0D3, APFloat(12.0), true},
  };
  for (auto i : tests) {
    auto f2i = this->Decode(i.f2i);
    auto i2f = this->Decode(i.i2f);
    ASSERT_TRUE(f2i.has_value());
    ASSERT_TRUE(i2f.has_value());
    uint32_t f_reg = DecodeRS1((*f2i).inst);
    uint32_t x_reg = DecodeRS1((*i2f).inst);
    if (i.isDouble)
      TestInverse<double>(this, f_reg, x_reg, *f2i, *i2f, i.data);
    else
      TestInverse<float>(this, f_reg, x_reg, *f2i, *i2f, i.data);
  }
}

TEST_F(RISCVEmulatorTester, TestFDInverse) {
  // FCVT_S_D
  auto d2f = this->Decode(0x4010F0D3);
  // FCVT_S_D
  auto f2d = this->Decode(0x4200F0D3);
  ASSERT_TRUE(d2f.has_value());
  ASSERT_TRUE(f2d.has_value());
  auto data = APFloat(12.0);
  uint32_t reg = DecodeRS1((*d2f).inst);
  float exp_f = 12.0f;
  double exp_d = 12.0;

  // double to float
  this->fpr.fpr[reg] = data.bitcastToAPInt().getZExtValue();
  ASSERT_TRUE(this->Execute(*d2f, false));
  ASSERT_EQ(this->fpr.fpr[reg], APFloat(exp_f).bitcastToAPInt().getZExtValue());

  // float to double
  ASSERT_TRUE(this->Execute(*f2d, false));
  ASSERT_EQ(this->fpr.fpr[reg], APFloat(exp_d).bitcastToAPInt().getZExtValue());
}

TEST_F(RISCVEmulatorTester, TestFloatLSInst) {
  uint32_t FLWInst = 0x1A207;  // imm = 0
  uint32_t FSWInst = 0x21A827; // imm = 16

  APFloat apf(12.0f);
  uint64_t bits = apf.bitcastToAPInt().getZExtValue();

  *(uint64_t *)this->memory = bits;
  auto decode = this->Decode(FLWInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FLW");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->fpr.fpr[DecodeRD(FLWInst)], bits);

  this->fpr.fpr[DecodeRS2(FSWInst)] = bits;
  decode = this->Decode(FSWInst);
  ASSERT_TRUE(decode.has_value());
  name = decode->pattern.name;
  ASSERT_EQ(name, "FSW");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(*(uint32_t *)(this->memory + 16), bits);
}

TEST_F(RISCVEmulatorTester, TestDoubleLSInst) {
  uint32_t FLDInst = 0x1B207;  // imm = 0
  uint32_t FSDInst = 0x21B827; // imm = 16

  APFloat apf(12.0);
  uint64_t bits = apf.bitcastToAPInt().getZExtValue();

  *(uint64_t *)this->memory = bits;
  auto decode = this->Decode(FLDInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FLD");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->fpr.fpr[DecodeRD(FLDInst)], bits);

  this->fpr.fpr[DecodeRS2(FSDInst)] = bits;
  decode = this->Decode(FSDInst);
  ASSERT_TRUE(decode.has_value());
  name = decode->pattern.name;
  ASSERT_EQ(name, "FSD");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(*(uint64_t *)(this->memory + 16), bits);
}

TEST_F(RISCVEmulatorTester, TestFMV_X_WInst) {
  auto FMV_X_WInst = 0xE0018253;

  APFloat apf(12.0f);
  auto exp_bits = apf.bitcastToAPInt().getZExtValue();
  this->fpr.fpr[DecodeRS1(FMV_X_WInst)] = NanBoxing(exp_bits);
  auto decode = this->Decode(FMV_X_WInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FMV_X_W");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->gpr.gpr[DecodeRD(FMV_X_WInst)], exp_bits);
}

TEST_F(RISCVEmulatorTester, TestFMV_X_DInst) {
  auto FMV_X_DInst = 0xE2018253;

  APFloat apf(12.0);
  auto exp_bits = apf.bitcastToAPInt().getZExtValue();
  this->fpr.fpr[DecodeRS1(FMV_X_DInst)] = exp_bits;
  auto decode = this->Decode(FMV_X_DInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FMV_X_D");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->gpr.gpr[DecodeRD(FMV_X_DInst)], exp_bits);
}

TEST_F(RISCVEmulatorTester, TestFMV_W_XInst) {
  auto FMV_W_XInst = 0xF0018253;

  APFloat apf(12.0f);
  uint64_t exp_bits = NanUnBoxing(apf.bitcastToAPInt().getZExtValue());
  this->gpr.gpr[DecodeRS1(FMV_W_XInst)] = exp_bits;
  auto decode = this->Decode(FMV_W_XInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FMV_W_X");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->fpr.fpr[DecodeRD(FMV_W_XInst)], exp_bits);
}

TEST_F(RISCVEmulatorTester, TestFMV_D_XInst) {
  auto FMV_D_XInst = 0xF2018253;

  APFloat apf(12.0);
  uint64_t bits = apf.bitcastToAPInt().getZExtValue();
  this->gpr.gpr[DecodeRS1(FMV_D_XInst)] = bits;
  auto decode = this->Decode(FMV_D_XInst);
  ASSERT_TRUE(decode.has_value());
  std::string name = decode->pattern.name;
  ASSERT_EQ(name, "FMV_D_X");
  ASSERT_TRUE(this->Execute(*decode, false));
  ASSERT_EQ(this->fpr.fpr[DecodeRD(FMV_D_XInst)], bits);
}
