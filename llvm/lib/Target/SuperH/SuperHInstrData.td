//===-- SuperHInstrData.td - SuperH Data Transfer Instructions -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes all of the data transfer instructions in the SuperH
/// ISA.
///
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

// imm -> Sign Extension -> Rn
class Move_ir<bits<16> op, string opcodestr>
	: SHInstOP_N4_I8<op, (outs GPR:$n), (ins imm8:$imm),
					 !strconcat(opcodestr, " $imm, $n")>;

// Rm -> Rn
class Move_rr<bits<16> op, string opcodestr>
	: SHInstOP_N4_M4<op, (outs GPR:$n), (ins GPR:$m),
					 !strconcat(opcodestr, " $m, $n")>;

// Rm -> (Rn)
class Move_rri<bits<16> op, string opcodestr>
	: SHInstOP_N4_M4<op, (outs GPR:$n), (ins GPR:$m),
					 !strconcat(opcodestr, " $m, $n")>;

// (Rm) -> Sign Extension -> Rn
class Move_rir<bits<16> op, string opcodestr>
	: SHInstOP_N4_M4<op, (outs GPR:$n), (ins GPR:$m),
					 !strconcat(opcodestr, " $m, $n")>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {
  def MOV_ir 		:  Move_ir<0b1110000000000000, "mov">;
  def MOV_rr 		:  Move_rr<0b0110000000000011, "mov">;
  def MOVB_rri	: Move_rri<0b0010000000000000, "mov.b">;
  def MOVW_rri	: Move_rri<0b0010000000000001, "mov.w">;
  def MOVL_rri	: Move_rri<0b0010000000000010, "mov.l">;
  def MOVB_rir	: Move_rir<0b0110000000000000, "mov.b">;
  def MOVW_rir	: Move_rir<0b0110000000000001, "mov.w">;
  def MOVL_rir	: Move_rir<0b0110000000000010, "mov.l">;
}