//===-- GenericOpcodes.td - Opcodes used with GlobalISel ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic opcodes used with GlobalISel.
// After instruction selection, these opcodes should not appear.
//
//===----------------------------------------------------------------------===//

//------------------------------------------------------------------------------
// Unary ops.
//------------------------------------------------------------------------------

// Extend the underlying scalar type of an operation, leaving the high bits
// unspecified.
def G_ANYEXT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

// Sign extend the underlying scalar type of an operation, copying the sign bit
// into the newly-created space.
def G_SEXT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

// Zero extend the underlying scalar type of an operation, putting zero bits
// into the newly-created space.
def G_ZEXT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}


// Truncate the underlying scalar type of an operation. This is equivalent to
// G_EXTRACT for scalar types, but acts elementwise on vectors.
def G_TRUNC : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_IMPLICIT_DEF : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins);
  let hasSideEffects = 0;
}

def G_PHI : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins variable_ops);
  let hasSideEffects = 0;
}

def G_FRAME_INDEX : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$src2);
  let hasSideEffects = 0;
}

def G_GLOBAL_VALUE : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

def G_INTTOPTR : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_PTRTOINT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_BITCAST : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_CONSTANT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$imm);
  let hasSideEffects = 0;
}

def G_FCONSTANT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$imm);
  let hasSideEffects = 0;
}

def G_VASTART : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins type0:$list);
  let hasSideEffects = 0;
  let mayStore = 1;
}

def G_VAARG : Instruction {
  let OutOperandList = (outs type0:$val);
  let InOperandList = (ins type1:$list, unknown:$align);
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 1;
}

def G_BSWAP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Binary ops.
//------------------------------------------------------------------------------

// Generic addition.
def G_ADD : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic subtraction.
def G_SUB : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic multiplication.
def G_MUL : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic signed division.
def G_SDIV : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic unsigned division.
def G_UDIV : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic signed remainder.
def G_SREM : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic unsigned remainder.
def G_UREM : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic bitwise and.
def G_AND : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic bitwise or.
def G_OR : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic bitwise xor.
def G_XOR : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic left-shift.
def G_SHL : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic logical right-shift.
def G_LSHR : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic arithmetic right-shift.
def G_ASHR : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic integer comparison.
def G_ICMP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$tst, type1:$src1, type1:$src2);
  let hasSideEffects = 0;
}

// Generic floating-point comparison.
def G_FCMP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins unknown:$tst, type1:$src1, type1:$src2);
  let hasSideEffects = 0;
}

// Generic select
def G_SELECT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$tst, type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic pointer offset.
def G_GEP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type1:$src2);
  let hasSideEffects = 0;
}

def G_PTR_MASK : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, unknown:$bits);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Overflow ops
//------------------------------------------------------------------------------

// Generic unsigned addition consuming and producing a carry flag.
def G_UADDE : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2, type1:$carry_in);
  let hasSideEffects = 0;
}

// Generic signed addition producing a carry flag.
def G_SADDO : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic unsigned subtraction consuming and producing a carry flag.
def G_USUBE : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2, type1:$carry_in);
  let hasSideEffects = 0;
}

// Generic unsigned subtraction producing a carry flag.
def G_SSUBO : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic unsigned multiplication producing a carry flag.
def G_UMULO : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic signed multiplication producing a carry flag.
def G_SMULO : Instruction {
  let OutOperandList = (outs type0:$dst, type1:$carry_out);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Multiply two numbers at twice the incoming bit width (unsigned) and return
// the high half of the result.
def G_UMULH : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Multiply two numbers at twice the incoming bit width (signed) and return
// the high half of the result.
def G_SMULH : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

//------------------------------------------------------------------------------
// Floating Point Unary Ops.
//------------------------------------------------------------------------------

def G_FNEG : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FPEXT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_FPTRUNC : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_FPTOSI : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_FPTOUI : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_SITOFP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

def G_UITOFP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Floating Point Binary ops.
//------------------------------------------------------------------------------

// Generic FP addition.
def G_FADD : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic FP subtraction.
def G_FSUB : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic FP multiplication.
def G_FMUL : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic fused multiply-add instruction.
// Behaves like llvm fma intrinsic ie src1 * src2 + src3
def G_FMA : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type0:$src3);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic FP division.
def G_FDIV : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Generic FP remainder.
def G_FREM : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Floating point exponentiation.
def G_FPOW : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = 0;
}

// Floating point base-e exponential of a value.
def G_FEXP : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = 0;
}

// Floating point base-2 exponential of a value.
def G_FEXP2 : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = 0;
}

// Floating point base-2 logarithm of a value.
def G_FLOG : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = 0;
}

// Floating point base-2 logarithm of a value.
def G_FLOG2 : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Memory ops
//------------------------------------------------------------------------------

// Generic load. Expects a MachineMemOperand in addition to explicit operands.
def G_LOAD : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins ptype1:$addr);
  let hasSideEffects = 0;
  let mayLoad = 1;
}

// Generic store. Expects a MachineMemOperand in addition to explicit operands.
def G_STORE : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins type0:$src, ptype1:$addr);
  let hasSideEffects = 0;
  let mayStore = 1;
}

//------------------------------------------------------------------------------
// Variadic ops
//------------------------------------------------------------------------------

// Extract a register of the specified size, starting from the block given by
// index. This will almost certainly be mapped to sub-register COPYs after
// register banks have been selected.
def G_EXTRACT : Instruction {
  let OutOperandList = (outs type0:$res);
  let InOperandList = (ins type1:$src, unknown:$offset);
  let hasSideEffects = 0;
}

// Extract multiple registers specified size, starting from blocks given by
// indexes. This will almost certainly be mapped to sub-register COPYs after
// register banks have been selected.
def G_UNMERGE_VALUES : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let hasSideEffects = 0;
}

// Insert a smaller register into a larger one at the specified bit-index.
def G_INSERT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$op, unknown:$offset);
  let hasSideEffects = 0;
}

/// Concatenante multiple registers of the same size into a wider register.
def G_MERGE_VALUES : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins variable_ops);
  let hasSideEffects = 0;
}

// Intrinsic without side effects.
def G_INTRINSIC : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins unknown:$intrin, variable_ops);
  let hasSideEffects = 0;
}

// Intrinsic with side effects.
def G_INTRINSIC_W_SIDE_EFFECTS : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins unknown:$intrin, variable_ops);
  let hasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 1;
}

//------------------------------------------------------------------------------
// Branches.
//------------------------------------------------------------------------------

// Generic unconditional branch.
def G_BR : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins unknown:$src1);
  let hasSideEffects = 0;
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
}

// Generic conditional branch.
def G_BRCOND : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins type0:$tst, unknown:$truebb);
  let hasSideEffects = 0;
  let isBranch = 1;
  let isTerminator = 1;
}

// Generic indirect branch.
def G_BRINDIRECT : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins type0:$src1);
  let hasSideEffects = 0;
  let isBranch = 1;
  let isTerminator = 1;
}

//------------------------------------------------------------------------------
// Vector ops
//------------------------------------------------------------------------------

// Generic insertelement.
def G_INSERT_VECTOR_ELT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$elt, type2:$idx);
  let hasSideEffects = 0;
}

// Generic extractelement.
def G_EXTRACT_VECTOR_ELT : Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, type2:$idx);
  let hasSideEffects = 0;
}

// Generic shufflevector.
def G_SHUFFLE_VECTOR: Instruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$v1, type1:$v2, type2:$mask);
  let hasSideEffects = 0;
}

// TODO: Add the other generic opcodes.
