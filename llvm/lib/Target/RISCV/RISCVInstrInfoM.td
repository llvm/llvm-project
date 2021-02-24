//===-- RISCVInstrInfoM.td - RISC-V 'M' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'M', Integer
// Multiplication and Division instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def riscv_divw  : SDNode<"RISCVISD::DIVW",  SDTIntBinOp>;
def riscv_divuw : SDNode<"RISCVISD::DIVUW", SDTIntBinOp>;
def riscv_remuw : SDNode<"RISCVISD::REMUW", SDTIntBinOp>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtM] in {
def MUL     : ALU_rr<0b0000001, 0b000, "mul">,
              Sched<[WriteIMul, ReadIMul, ReadIMul]>;
def MULH    : ALU_rr<0b0000001, 0b001, "mulh">,
              Sched<[WriteIMul, ReadIMul, ReadIMul]>;
def MULHSU  : ALU_rr<0b0000001, 0b010, "mulhsu">,
              Sched<[WriteIMul, ReadIMul, ReadIMul]>;
def MULHU   : ALU_rr<0b0000001, 0b011, "mulhu">,
              Sched<[WriteIMul, ReadIMul, ReadIMul]>;
def DIV     : ALU_rr<0b0000001, 0b100, "div">,
              Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
def DIVU    : ALU_rr<0b0000001, 0b101, "divu">,
              Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
def REM     : ALU_rr<0b0000001, 0b110, "rem">,
              Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
def REMU    : ALU_rr<0b0000001, 0b111, "remu">,
              Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
} // Predicates = [HasStdExtM]

let Predicates = [HasStdExtM, IsRV64] in {
def MULW    : ALUW_rr<0b0000001, 0b000, "mulw">,
              Sched<[WriteIMul32, ReadIMul32, ReadIMul32]>;
def DIVW    : ALUW_rr<0b0000001, 0b100, "divw">,
              Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>;
def DIVUW   : ALUW_rr<0b0000001, 0b101, "divuw">,
              Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>;
def REMW    : ALUW_rr<0b0000001, 0b110, "remw">,
              Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>;
def REMUW   : ALUW_rr<0b0000001, 0b111, "remuw">,
              Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>;
} // Predicates = [HasStdExtM, IsRV64]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtM] in {
def : PatGprGpr<mul, MUL>;
def : PatGprGpr<mulhs, MULH>;
def : PatGprGpr<mulhu, MULHU>;
// No ISDOpcode for mulhsu
def : PatGprGpr<sdiv, DIV>;
def : PatGprGpr<udiv, DIVU>;
def : PatGprGpr<srem, REM>;
def : PatGprGpr<urem, REMU>;
} // Predicates = [HasStdExtM]

let Predicates = [HasStdExtM, IsRV64] in {
def : Pat<(sext_inreg (mul GPR:$rs1, GPR:$rs2), i32),
          (MULW GPR:$rs1, GPR:$rs2)>;

def : PatGprGpr<riscv_divw, DIVW>;
def : PatGprGpr<riscv_divuw, DIVUW>;
def : PatGprGpr<riscv_remuw, REMUW>;

// Handle the specific cases where using DIVU/REMU would be correct and result
// in fewer instructions than emitting DIVUW/REMUW then zero-extending the
// result.
def : Pat<(and (riscv_divuw (assertzexti32 GPR:$rs1),
                            (assertzexti32 GPR:$rs2)), 0xffffffff),
          (DIVU GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (riscv_remuw (assertzexti32 GPR:$rs1),
                            (assertzexti32 GPR:$rs2)), 0xffffffff),
          (REMU GPR:$rs1, GPR:$rs2)>;

// Although the sexti32 operands may not have originated from an i32 srem,
// this pattern is safe as it is impossible for two sign extended inputs to
// produce a result where res[63:32]=0 and res[31]=1.
def : Pat<(srem (sexti32 (i64 GPR:$rs1)), (sexti32 (i64 GPR:$rs2))),
          (REMW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtM, IsRV64]
