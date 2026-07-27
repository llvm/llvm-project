//===-- RISCVInstrInfoZilx.td - 'Zilx' instructions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zilx' (Indexed Integer
/// Load).
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

// Zilx indexed loads reuse the AMO major opcode. The addressing mode is
// selected by funct5 (inst[31:27]) and the access width/signedness reuses the
// base integer load funct3. rs1 is the index operand, rs2 is the base operand,
// and the aq and rl bits are always 0. The base register is written in
// parentheses to distinguish it from the index register.
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class ZilxLoad<bits<5> funct5, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<funct5, /*aq=*/0, /*rl=*/0, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins GPRMemZeroOffset:$rs2, GPR:$rs1),
                    opcodestr, "$rd, $rs2, $rs1">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilx] in {
// Unscaled indexed loads: address = base + index. Byte loads are not provided
// in the unscaled mode.
def LXH  : ZilxLoad<0b10010, 0b001, "lxh">,
           Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXW  : ZilxLoad<0b10010, 0b010, "lxw">,
           Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXHU : ZilxLoad<0b10010, 0b101, "lxhu">,
           Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads: address = base + (index << log2(access-size)). Byte
// loads are provided only in scaled form (the byte scale factor is 1).
def LXSB  : ZilxLoad<0b11010, 0b000, "lxsb">,
            Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSH  : ZilxLoad<0b11010, 0b001, "lxsh">,
            Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSW  : ZilxLoad<0b11010, 0b010, "lxsw">,
            Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXSBU : ZilxLoad<0b11010, 0b100, "lxsbu">,
            Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSHU : ZilxLoad<0b11010, 0b101, "lxshu">,
            Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
} // Predicates = [HasStdExtZilx]

let Predicates = [HasStdExtZilx, IsRV64] in {
// Unscaled indexed loads (RV64-only widths).
def LXD  : ZilxLoad<0b10010, 0b011, "lxd">,
           Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXWU : ZilxLoad<0b10010, 0b110, "lxwu">,
           Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads (RV64-only widths).
def LXSD  : ZilxLoad<0b11010, 0b011, "lxsd">,
            Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXSWU : ZilxLoad<0b11010, 0b110, "lxswu">,
            Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads with a zero-extended 32-bit index (RV64-only):
// address = base + (zext32(index) << log2(access-size)).
def LXSUWB  : ZilxLoad<0b11110, 0b000, "lxsuwb">,
              Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSUWH  : ZilxLoad<0b11110, 0b001, "lxsuwh">,
              Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSUWW  : ZilxLoad<0b11110, 0b010, "lxsuww">,
              Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXSUWD  : ZilxLoad<0b11110, 0b011, "lxsuwd">,
              Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXSUWBU : ZilxLoad<0b11110, 0b100, "lxsuwbu">,
              Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSUWHU : ZilxLoad<0b11110, 0b101, "lxsuwhu">,
              Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSUWWU : ZilxLoad<0b11110, 0b110, "lxsuwwu">,
              Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
} // Predicates = [HasStdExtZilx, IsRV64]

// `lxb`/`lxbu` are assembler pseudoinstructions for the scaled byte loads.
// They are parse-only aliases (EmitPriority 0) so the disassembler still prints
// the underlying `lxsb`/`lxsbu`.
let Predicates = [HasStdExtZilx] in {
def : InstAlias<"lxb $rd, $rs2, $rs1",  (LXSB  GPR:$rd, GPRMemZeroOffset:$rs2, GPR:$rs1), 0>;
def : InstAlias<"lxbu $rd, $rs2, $rs1", (LXSBU GPR:$rd, GPRMemZeroOffset:$rs2, GPR:$rs1), 0>;
}
