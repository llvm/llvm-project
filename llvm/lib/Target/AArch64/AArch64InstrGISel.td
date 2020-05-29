//=----- AArch64InstrGISel.td - AArch64 GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AArch64 GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//


class AArch64GenericInstruction : GenericInstruction {
  let Namespace = "AArch64";
}

// A pseudo to represent a relocatable add instruction as part of address
// computation.
def G_ADD_LOW : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, type2:$imm);
  let hasSideEffects = 0;
}
