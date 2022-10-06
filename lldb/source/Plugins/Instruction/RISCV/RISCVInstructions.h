//===-- RISCVInstructions.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_RISCVINSTRUCTION_H
#define LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_RISCVINSTRUCTION_H

#include <cstdint>
#include <variant>

#include "EmulateInstructionRISCV.h"
#include "llvm/ADT/Optional.h"

namespace lldb_private {

class EmulateInstructionRISCV;

struct Rd {
  uint32_t rd;
  bool Write(EmulateInstructionRISCV &emulator, uint64_t value);
};

struct Rs {
  uint32_t rs;
  llvm::Optional<uint64_t> Read(EmulateInstructionRISCV &emulator);
  llvm::Optional<int32_t> ReadI32(EmulateInstructionRISCV &emulator);
  llvm::Optional<int64_t> ReadI64(EmulateInstructionRISCV &emulator);
  llvm::Optional<uint32_t> ReadU32(EmulateInstructionRISCV &emulator);
};

#define I_TYPE_INST(NAME)                                                      \
  struct NAME {                                                                \
    Rd rd;                                                                     \
    Rs rs1;                                                                    \
    uint32_t imm;                                                              \
  }
#define S_TYPE_INST(NAME)                                                      \
  struct NAME {                                                                \
    Rs rs1;                                                                    \
    Rs rs2;                                                                    \
    uint32_t imm;                                                              \
  }
#define U_TYPE_INST(NAME)                                                      \
  struct NAME {                                                                \
    Rd rd;                                                                     \
    uint32_t imm;                                                              \
  }
/// The memory layout are the same in our code.
#define J_TYPE_INST(NAME) U_TYPE_INST(NAME)
#define R_TYPE_INST(NAME)                                                      \
  struct NAME {                                                                \
    Rd rd;                                                                     \
    Rs rs1;                                                                    \
    Rs rs2;                                                                    \
  }
#define R_SHAMT_TYPE_INST(NAME)                                                \
  struct NAME {                                                                \
    Rd rd;                                                                     \
    Rs rs1;                                                                    \
    uint32_t shamt;                                                            \
  }
#define R_RS1_TYPE_INST(NAME)                                                  \
  struct NAME {                                                                \
    Rd rd;                                                                     \
    Rs rs1;                                                                    \
  }

// RV32I instructions (The base integer ISA)
struct B {
  Rs rs1;
  Rs rs2;
  uint32_t imm;
  uint32_t funct3;
};
U_TYPE_INST(LUI);
U_TYPE_INST(AUIPC);
J_TYPE_INST(JAL);
I_TYPE_INST(JALR);
I_TYPE_INST(LB);
I_TYPE_INST(LH);
I_TYPE_INST(LW);
I_TYPE_INST(LBU);
I_TYPE_INST(LHU);
S_TYPE_INST(SB);
S_TYPE_INST(SH);
S_TYPE_INST(SW);
I_TYPE_INST(ADDI);
I_TYPE_INST(SLTI);
I_TYPE_INST(SLTIU);
I_TYPE_INST(XORI);
I_TYPE_INST(ORI);
I_TYPE_INST(ANDI);
R_TYPE_INST(ADD);
R_TYPE_INST(SUB);
R_TYPE_INST(SLL);
R_TYPE_INST(SLT);
R_TYPE_INST(SLTU);
R_TYPE_INST(XOR);
R_TYPE_INST(SRL);
R_TYPE_INST(SRA);
R_TYPE_INST(OR);
R_TYPE_INST(AND);

// RV64I inst (The base integer ISA)
I_TYPE_INST(LWU);
I_TYPE_INST(LD);
S_TYPE_INST(SD);
R_SHAMT_TYPE_INST(SLLI);
R_SHAMT_TYPE_INST(SRLI);
R_SHAMT_TYPE_INST(SRAI);
I_TYPE_INST(ADDIW);
R_SHAMT_TYPE_INST(SLLIW);
R_SHAMT_TYPE_INST(SRLIW);
R_SHAMT_TYPE_INST(SRAIW);
R_TYPE_INST(ADDW);
R_TYPE_INST(SUBW);
R_TYPE_INST(SLLW);
R_TYPE_INST(SRLW);
R_TYPE_INST(SRAW);

// RV32M inst (The standard integer multiplication and division extension)
R_TYPE_INST(MUL);
R_TYPE_INST(MULH);
R_TYPE_INST(MULHSU);
R_TYPE_INST(MULHU);
R_TYPE_INST(DIV);
R_TYPE_INST(DIVU);
R_TYPE_INST(REM);
R_TYPE_INST(REMU);

// RV64M inst (The standard integer multiplication and division extension)
R_TYPE_INST(MULW);
R_TYPE_INST(DIVW);
R_TYPE_INST(DIVUW);
R_TYPE_INST(REMW);
R_TYPE_INST(REMUW);

// RV32A inst (The standard atomic instruction extension)
R_RS1_TYPE_INST(LR_W);
R_TYPE_INST(SC_W);
R_TYPE_INST(AMOSWAP_W);
R_TYPE_INST(AMOADD_W);
R_TYPE_INST(AMOXOR_W);
R_TYPE_INST(AMOAND_W);
R_TYPE_INST(AMOOR_W);
R_TYPE_INST(AMOMIN_W);
R_TYPE_INST(AMOMAX_W);
R_TYPE_INST(AMOMINU_W);
R_TYPE_INST(AMOMAXU_W);

// RV64A inst (The standard atomic instruction extension)
R_RS1_TYPE_INST(LR_D);
R_TYPE_INST(SC_D);
R_TYPE_INST(AMOSWAP_D);
R_TYPE_INST(AMOADD_D);
R_TYPE_INST(AMOXOR_D);
R_TYPE_INST(AMOAND_D);
R_TYPE_INST(AMOOR_D);
R_TYPE_INST(AMOMIN_D);
R_TYPE_INST(AMOMAX_D);
R_TYPE_INST(AMOMINU_D);
R_TYPE_INST(AMOMAXU_D);

using RISCVInst =
    std::variant<LUI, AUIPC, JAL, JALR, B, LB, LH, LW, LBU, LHU, SB, SH, SW,
                 ADDI, SLTI, SLTIU, XORI, ORI, ANDI, ADD, SUB, SLL, SLT, SLTU,
                 XOR, SRL, SRA, OR, AND, LWU, LD, SD, SLLI, SRLI, SRAI, ADDIW,
                 SLLIW, SRLIW, SRAIW, ADDW, SUBW, SLLW, SRLW, SRAW, MUL, MULH,
                 MULHSU, MULHU, DIV, DIVU, REM, REMU, MULW, DIVW, DIVUW, REMW,
                 REMUW, LR_W, SC_W, AMOSWAP_W, AMOADD_W, AMOXOR_W, AMOAND_W,
                 AMOOR_W, AMOMIN_W, AMOMAX_W, AMOMINU_W, AMOMAXU_W, LR_D, SC_D,
                 AMOSWAP_D, AMOADD_D, AMOXOR_D, AMOAND_D, AMOOR_D, AMOMIN_D,
                 AMOMAX_D, AMOMINU_D, AMOMAXU_D>;

struct InstrPattern {
  const char *name;
  /// Bit mask to check the type of a instruction (B-Type, I-Type, J-Type, etc.)
  uint32_t type_mask;
  /// Characteristic value after bitwise-and with type_mask.
  uint32_t eigen;
  RISCVInst (*decode)(uint32_t inst);
};

struct DecodeResult {
  RISCVInst decoded;
  uint32_t inst;
  bool is_rvc;
  InstrPattern pattern;
};

constexpr uint32_t DecodeRD(uint32_t inst) { return (inst & 0xF80) >> 7; }
constexpr uint32_t DecodeRS1(uint32_t inst) { return (inst & 0xF8000) >> 15; }
constexpr uint32_t DecodeRS2(uint32_t inst) { return (inst & 0x1F00000) >> 20; }

// decode register for RVC
constexpr uint16_t DecodeCR_RD(uint16_t inst) { return DecodeRD(inst); }
constexpr uint16_t DecodeCI_RD(uint16_t inst) { return DecodeRD(inst); }
constexpr uint16_t DecodeCIW_RD(uint16_t inst) { return (inst & 0x1C) >> 2; }
constexpr uint16_t DecodeCL_RD(uint16_t inst) { return DecodeCIW_RD(inst); }
constexpr uint16_t DecodeCA_RD(uint16_t inst) { return (inst & 0x380) >> 7; }
constexpr uint16_t DecodeCB_RD(uint16_t inst) { return DecodeCA_RD(inst); }

constexpr uint16_t DecodeCR_RS1(uint16_t inst) { return DecodeRD(inst); }
constexpr uint16_t DecodeCI_RS1(uint16_t inst) { return DecodeRD(inst); }
constexpr uint16_t DecodeCL_RS1(uint16_t inst) { return DecodeCA_RD(inst); }
constexpr uint16_t DecodeCS_RS1(uint16_t inst) { return DecodeCA_RD(inst); }
constexpr uint16_t DecodeCA_RS1(uint16_t inst) { return DecodeCA_RD(inst); }
constexpr uint16_t DecodeCB_RS1(uint16_t inst) { return DecodeCA_RD(inst); }

constexpr uint16_t DecodeCR_RS2(uint16_t inst) { return (inst & 0x7C) >> 2; }
constexpr uint16_t DecodeCSS_RS2(uint16_t inst) { return DecodeCR_RS2(inst); }
constexpr uint16_t DecodeCS_RS2(uint16_t inst) { return DecodeCIW_RD(inst); }
constexpr uint16_t DecodeCA_RS2(uint16_t inst) { return DecodeCIW_RD(inst); }

} // namespace lldb_private
#endif // LLDB_SOURCE_PLUGINS_INSTRUCTION_RISCV_RISCVINSTRUCTION_H
