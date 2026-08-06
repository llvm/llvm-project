//===-- SuperHInstrBranch.td - SuperH Branching Instructions -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//
///
/// \file
/// This file describes all of the arithmetic instructions in the SuperH
/// ISA.
///
//===-----------------------------------------------------------------------===//

//===-----------------------------------------------------------------------===//
// Instruction Class Templates
//===-----------------------------------------------------------------------===//


// disp + PC -> PC
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = true in
class BrOp_Disp8<bits<16> op, string opcodestr> 
      : SHInstOP_D8<op, (outs), (ins disp8:$disp),
                    !strconcat(opcodestr, " $disp")>;
// disp + PC -> PC
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = true in
class BrOp_Disp12<bits<16> op, string opcodestr> 
      : SHInstOP_D12<op, (outs), (ins disp12:$disp),
                    !strconcat(opcodestr, " $disp")>;

// disp + PC -> PC (Delayed)
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = true, hasDelaySlot = true in
class BrOpD_Disp8<bits<16> op, string opcodestr> 
      : SHInstOP_D8<op, (outs), (ins disp8:$disp),
                    !strconcat(opcodestr, " $disp")>;

// Rm + PC -> PC
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = true, isIndirectBranch = true in
class BrOp_Rm<bits<16> op, string opcodestr> 
      : SHInstOP_M4<op, (outs), (ins GPR:$Rm),
                    !strconcat(opcodestr, " $Rm")>;
// Rm -> PC
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = true, isIndirectBranch = true in
class BrOp_Rmi<bits<16> op, string opcodestr> 
      : SHInstOP_M4<op, (outs), (ins GPRMem:$Rm),
                    !strconcat(opcodestr, " @$Rm")>;

// CALL
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCall = true in
class BrOpCall_Rmi<bits<16> op, string opcodestr> 
      : SHInstOP_M4<op, (outs), (ins GPRMem:$Rm),
                    !strconcat(opcodestr, " @$Rm")>;

// RETURN
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isReturn = true in
class BrOpRet<bits<16> op, string opcodestr> 
      : SHInstOP<op, (outs), (ins), opcodestr>;


//===-----------------------------------------------------------------------===//
// Branch Instructions
//===-----------------------------------------------------------------------===//

let Namespace = "SH" in {
	def BF_Disp			 : BrOp_Disp8<0b1000101100000000, "bf">;
	def BFS_Disp	  : BrOpD_Disp8<0b1000111100000000, "bf/s">;
	def BT_Disp			 : BrOp_Disp8<0b1000100100000000, "bt">;
	def BTS_Disp	  : BrOpD_Disp8<0b1000110100000000, "bt/s">;
	def BRA_Disp		: BrOp_Disp12<0b1010000000000000, "bra">;
	def BRAF_Rm 	      : BrOp_Rm<0b0000000000100011, "braf">;
	def BSR_Disp		: BrOp_Disp12<0b1011000000000000, "bsr">;
	def BSRF_Rm 	      : BrOp_Rm<0b0000000000000011, "bsrf">;
	def JMP_Rmi        : BrOp_Rmi<0b0100000000101011, "jmp">;
	def JSR_Rmi    : BrOpCall_Rmi<0b0100000000001011, "jsr">;
	def RTS 						: BrOpRet<0b0000000000001011, "rts">;
}