//=----- RISCVInstrGISel.td - RISCV GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCV GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//

class RISCVGenericInstruction : GenericInstruction {
  let Namespace = "RISCV";
}

// A pseudo to represent a relocatable add instruction as part of address
// computation.
// FIXME: Copy the type from SelectionDAG
def G_RISCV_ADD_LO : RISCVGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, type2:$imm);
  let hasSideEffects = 0;
}

// FIXME: Copy the type from SelectionDAG
def G_RISCV_HI : RISCVGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$imm);
  let hasSideEffects = 0;
}

def : GINodeEquiv<G_RISCV_ADD_LO, riscv_add_lo>;
def : GINodeEquiv<G_RISCV_HI, riscv_hi>;
