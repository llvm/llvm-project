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
def G_ANYEXTEND : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

// Sign extend the underlying scalar type of an operation, copying the sign bit
// into the newly-created space.
def G_SEXT : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

// Zero extend the underlying scalar type of an operation, putting zero bits
// into the newly-created space.
def G_ZEXT : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}


// Truncate the underlying scalar type of an operation. This is equivalent to
// G_EXTRACT for scalar types, but acts elementwise on vectors.
def G_TRUNC : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

def G_FRAME_INDEX : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src2);
  let hasSideEffects = 0;
}

def G_INTTOPTR : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

def G_PTRTOINT : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

def G_BITCAST : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let hasSideEffects = 0;
}

def G_CONSTANT : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$imm);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Binary ops.
//------------------------------------------------------------------------------

// Generic addition.
def G_ADD : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic subtraction.
def G_SUB : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 0;
}

// Generic multiplication.
def G_MUL : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic addition consuming and producing a carry flag.
def G_ADDE : Instruction {
  let OutOperandList = (outs unknown:$dst, unknown:$carry_out);
  let InOperandList = (ins unknown:$src1, unknown:$src2, unknown:$carry_in);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic bitwise and.
def G_AND : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic bitwise or.
def G_OR : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic bitwise xor.
def G_XOR : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
  let isCommutable = 1;
}

// Generic left-shift.
def G_SHL : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
}

// Generic logical right-shift.
def G_LSHR : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
}

// Generic arithmetic right-shift.
def G_ASHR : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src1, unknown:$src2);
  let hasSideEffects = 0;
}

//------------------------------------------------------------------------------
// Memory ops
//------------------------------------------------------------------------------

// Generic load. Expects a MachineMemOperand in addition to explicit operands.
def G_LOAD : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$addr);
  let hasSideEffects = 0;
  let mayLoad = 1;
}

// Generic store. Expects a MachineMemOperand in addition to explicit operands.
def G_STORE : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins unknown:$src, unknown:$addr);
  let hasSideEffects = 0;
  let mayStore = 1;
}

//------------------------------------------------------------------------------
// Variadic ops
//------------------------------------------------------------------------------

// Extract multiple registers specified size, starting from blocks given by
// indexes. This will almost certainly be mapped to sub-register COPYs after
// register banks have been selected.
def G_EXTRACT : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let hasSideEffects = 0;
}

// Combine a sequence of generic vregs into a single larger value (starting at
// bit 0).
def G_SEQUENCE : Instruction {
  let OutOperandList = (outs unknown:$dst);
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
  let InOperandList = (ins unknown:$tst, unknown:$truebb);
  let hasSideEffects = 0;
  let isBranch = 1;
  let isTerminator = 1;
}

// TODO: Add the other generic opcodes.
