//===-- RISCVInstrInfoZicbo.td - RISC-V CMO instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Base Cache
// Management Operation ISA Extensions document (Zicbop, Zicboz, and Zicbop).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CBO_r<bits<12> optype, string opcodestr>
    : RVInstI<0b010, OPC_MISC_MEM, (outs), (ins GPRMemZeroOffset:$rs1),
              opcodestr, "$rs1"> {
  let imm12 = optype;
  let rd = 0b00000;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicbom] in {
def CBO_CLEAN : CBO_r<0b000000000001, "cbo.clean">, Sched<[]>;
def CBO_FLUSH : CBO_r<0b000000000010, "cbo.flush">, Sched<[]>;
def CBO_INVAL : CBO_r<0b000000000000, "cbo.inval">, Sched<[]>;
} // Predicates = [HasStdExtZicbom]

let Predicates = [HasStdExtZicboz] in {
def CBO_ZERO : CBO_r<0b000000000100, "cbo.zero">, Sched<[]>;
} // Predicates = [HasStdExtZicboz]
