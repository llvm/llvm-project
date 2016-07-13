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
// Binary ops.
//------------------------------------------------------------------------------
// Generic addition.
def G_ADD : Instruction {
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
