//===-- VIInstructions.td - VI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for VI and newer.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// SMEM Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isVI] in {

// 1. Offset as 20bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM20bit:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_i32imm $offset))
>;

} // End Predicates = [isVI]
