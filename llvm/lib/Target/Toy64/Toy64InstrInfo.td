//===-- Toy64InstrInfo.td - Toy64 Instruction Definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Toy64 machine instructions.
//
//===----------------------------------------------------------------------===//

include "Toy64InstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand definitions
//===----------------------------------------------------------------------===//

// A signed 12-bit immediate.
//
// The MachineInstr representation stores the value as an i64 operand.
// The instruction encoding only contains the low 12 encoded bits.
//
// We will add proper range checking when the assembler/parser and
// instruction-selection patterns are introduced.
def simm12 : Operand<i64>;

//===----------------------------------------------------------------------===//
// Integer arithmetic instructions
//===----------------------------------------------------------------------===//

// ADD rd, rs1, rs2
//
// rd = rs1 + rs2
//
let isCommutable = 1 in
def ADD : Toy64RType<
    0b0000000,                // funct7
    0b000,                    // funct3
    0b0110011,                // opcode
    (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2),
    "add\t$rd, $rs1, $rs2"
>;

// SUB rd, rs1, rs2
//
// rd = rs1 - rs2
//
def SUB : Toy64RType<
    0b0100000,                // funct7
    0b000,                    // funct3
    0b0110011,                // opcode
    (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2),
    "sub\t$rd, $rs1, $rs2"
>;

// ADDI rd, rs1, imm12
//
// rd = rs1 + sign_extend(imm12)
//
def ADDI : Toy64IType<
    0b000,                    // funct3
    0b0010011,                // opcode
    (outs GPR:$rd),
    (ins GPR:$rs1, simm12:$imm12),
    "addi\t$rd, $rs1, $imm12"
>;