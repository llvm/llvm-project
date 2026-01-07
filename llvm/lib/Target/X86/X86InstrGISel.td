//===- X86InstrGISel.td - X86 GISel target specific opcodes -*- tablegen -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// X86 GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//

class X86GenericInstruction : GenericInstruction { let Namespace = "X86"; }

def G_FILD : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins ptype1:$src);
  let hasSideEffects = false;
  let mayLoad = true;
}
def G_FIST : X86GenericInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins type0:$src1, ptype1:$src2);
  let hasSideEffects = false;
  let mayStore = true;
}

def G_FNSTCW16 : X86GenericInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins ptype0:$dst);
  let hasSideEffects = true;
  let mayStore = true;
}

def G_FLDCW16 : X86GenericInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins ptype0:$src);
  let hasSideEffects = true;
  let mayLoad = true;
}

def : GINodeEquiv<G_FILD, X86fild>;
def : GINodeEquiv<G_FIST, X86fp_to_mem>;
def : GINodeEquiv<G_FNSTCW16, X86fp_cwd_get16>;
def : GINodeEquiv<G_FLDCW16, X86fp_cwd_set16>;
