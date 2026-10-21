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
// Toy64-specific SelectionDAG nodes
//===----------------------------------------------------------------------===//

// Target-specific return node.
//
// LowerReturn() will create Toy64ISD::RET_GLUE after copying the return
// value into the ABI return register (X10).
//
// The node carries a chain because returning changes control flow.
// It may also consume glue from COPY_TO_REG.
def toy64_ret_glue :
    SDNode<"Toy64ISD::RET_GLUE",
           SDTNone,
           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

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
def simm12 : Operand<i64>,ImmLeaf<i64, [{return isInt<12>(Imm);}]>;


// SelectionDAG address matcher.
//
// SelectAddr() will decompose an address into:
//
//     Base, Offset
//
def addr : ComplexPattern<i64, 2, "SelectAddr",[], []>;

//===----------------------------------------------------------------------===//
// Call-frame pseudo instructions
//===----------------------------------------------------------------------===//

let Defs = [X2],
    Uses = [X2],
    hasSideEffects = 1 in {

def ADJCALLSTACKDOWN :
    Toy64Pseudo<
        (outs),
        (ins i64imm:$amt1, i64imm:$amt2),
        "# ADJCALLSTACKDOWN $amt1, $amt2"
    >;

def ADJCALLSTACKUP :
    Toy64Pseudo<
        (outs),
        (ins i64imm:$amt1, i64imm:$amt2),
        "# ADJCALLSTACKUP $amt1, $amt2"
    >;

}

//===----------------------------------------------------------------------===//
// Integer arithmetic instructions
//===----------------------------------------------------------------------===//

// ADD rd, rs1, rs2
//
// rd = rs1 + rs2
//
let isCommutable = 1 in
def ADD : Toy64RType<
    0b0000000,
    0b000,
    0b0110011,
    (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2),
    "add\t$rd, $rs1, $rs2",
    [(set GPR:$rd,
          (add GPR:$rs1, GPR:$rs2))]
>;

// SUB rd, rs1, rs2
//
// rd = rs1 - rs2
//
def SUB : Toy64RType<
    0b0100000,
    0b000,
    0b0110011,
    (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2),
    "sub\t$rd, $rs1, $rs2",
    [(set GPR:$rd,
          (sub GPR:$rs1, GPR:$rs2))]
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

//===----------------------------------------------------------------------===//
// Integer code-generation patterns
//===----------------------------------------------------------------------===//

// Materialize a small signed constant:
//
//     imm
//
// becomes:
//
//     ADDI rd, X0, imm
//
def : Pat<(simm12:$imm), (ADDI X0, simm12:$imm)>;

// Register + signed 12-bit immediate:
//
//     add rs1, imm
//
// becomes:
//
//     ADDI rd, rs1, imm
//
def : Pat< (add GPR:$rs1, simm12:$imm12), (ADDI GPR:$rs1, simm12:$imm12)>;


//===----------------------------------------------------------------------===//
// Memory instructions
//===----------------------------------------------------------------------===//

// LD rd, rs1, imm12
//
// rd = Mem[rs1 + sign_extend(imm12)]
//
let mayLoad = 1 in
def LD : Toy64IType<
    0b011,                    // funct3: 64-bit load
    0b0000011,                // opcode
    (outs GPR:$rd),
    (ins GPR:$rs1, simm12:$imm12),
    "ld\t$rd, $imm12($rs1)",
    [(set GPR:$rd,(load (addr GPR:$rs1,simm12:$imm12)))]>;

// SD rs2, rs1, imm12
//
// Mem[rs1 + sign_extend(imm12)] = rs2
//
let mayStore = 1 in
def SD : Toy64SType<
    0b011,                    // funct3: 64-bit store
    0b0100011,                // opcode
    (outs),
    (ins GPR:$rs2,
         GPR:$rs1,
         simm12:$imm12),
    "sd\t$rs2, $imm12($rs1)",
    [(store GPR:$rs2,(addr GPR:$rs1,simm12:$imm12))]
>;

//===----------------------------------------------------------------------===//
// Return
//===----------------------------------------------------------------------===//

// RET is encoded as:
//
//     JALR X0, X1, 0
//
// rd    = X0 = 0
// rs1   = X1 = 1
// imm12 = 0
//
// opcode = 1100111
// funct3 = 000
//
let isBarrier = 1,
    isReturn = 1,
    isTerminator = 1,
    Uses = [X1] in
def RET : Toy64IType<
    0b000,
    0b1100111,
    (outs),
    (ins),
    "ret",
    [(toy64_ret_glue)]
> {
  let rd = 0;
  let rs1 = 1;
  let imm12 = 0;
}