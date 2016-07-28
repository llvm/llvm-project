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
}

// TODO: Add the other generic opcodes.
