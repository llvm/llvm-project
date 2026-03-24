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

// X86-specific shuffle operations 

// PSHUFB - Packed Shuffle Bytes (SSSE3+)
// Byte-level shuffle using control mask from second operand
def G_X86_PSHUFB : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type0:$mask);
  let hasSideEffects = false;
}

// PSHUFD/PSHUFLW/PSHUFHW - Shuffle doublewords/low words/high words
// In-lane shuffle with immediate control
def G_X86_PSHUFD : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, untyped_imm_0:$imm);
  let hasSideEffects = false;
}

// SHUFPS/SHUFPD - Shuffle packed single/double precision floats
def G_X86_SHUFP : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, untyped_imm_0:$imm);
  let hasSideEffects = false;
}

// UNPCKL/UNPCKH - Unpack low/high data
def G_X86_UNPCKL : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}

def G_X86_UNPCKH : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}

// VBROADCAST - Broadcast a single element to all elements
def G_X86_VBROADCAST : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = false;
}

// VPERMILPS/VPERMILPD - Permute in-lane with variable control (AVX+)
def G_X86_VPERMILPV : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type0:$mask);
  let hasSideEffects = false;
}

// VPERMILPS/VPERMILPD - Permute in-lane with immediate control (AVX+)
def G_X86_VPERMILPI : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, untyped_imm_0:$imm);
  let hasSideEffects = false;
}

// VPERMQ/VPERMPD - Permute 256-bit lanes (AVX2+)
def G_X86_VPERMI : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, untyped_imm_0:$imm);
  let hasSideEffects = false;
}

// BLENDVPS/BLENDVPD/PBLENDVB - Variable blend using mask register (SSE4.1+)
def G_X86_BLENDV : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type0:$mask);
  let hasSideEffects = false;
}

def : GINodeEquiv<G_FILD, X86fild>;
def : GINodeEquiv<G_FIST, X86fp_to_mem>;
def : GINodeEquiv<G_FNSTCW16, X86fp_cwd_get16>;
def : GINodeEquiv<G_FLDCW16, X86fp_cwd_set16>;

// X86 Shuffle Node Equivalences
def : GINodeEquiv<G_X86_PSHUFB, X86pshufb>;
def : GINodeEquiv<G_X86_SHUFP, X86Shufp>;
def : GINodeEquiv<G_X86_UNPCKL, X86Unpckl>;
def : GINodeEquiv<G_X86_UNPCKH, X86Unpckh>;
def : GINodeEquiv<G_X86_VBROADCAST, X86VBroadcast>;
def : GINodeEquiv<G_X86_VPERMILPV, X86VPermilpv>;
def : GINodeEquiv<G_X86_VPERMILPI, X86VPermilpi>;
def : GINodeEquiv<G_X86_VPERMI, X86VPermi>;
def : GINodeEquiv<G_X86_BLENDV, X86Blendv>;
