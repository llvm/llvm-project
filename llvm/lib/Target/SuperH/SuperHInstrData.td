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

// LOAD
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {

  // X -> Rn
  class LdOp_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4<op, (outs GPR:$Rn), (ins),
                      !strconcat(opcodestr, " $Rn")>;

  // Rm -> Rn
  class LdOp_Rm_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPR:$Rm),
                         !strconcat(opcodestr, " $Rm, $Rn")>;

  // #imm -> sign extension -> Rn
  class LdOp_Imm_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_I8<op, (outs GPR:$Rn), (ins imm8:$imm),
                         !strconcat(opcodestr, " $imm, $Rn")>;

  // (disp) -> R0
  class LdOp_DispPC_R0<bits<16> op, string opcodestr> 
        : SHInstOP_D8<op, (outs), (ins disp8:$disp),
                         !strconcat(opcodestr, " @($disp,pc),R0")>;

  // (disp) -> [sign extension] -> Rn
  class LdOp_DispPC_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_D8<op, (outs GPR:$Rn), (ins disp8:$disp),
                         !strconcat(opcodestr, " @($disp,pc), $Rn")>;

  // (Rm) -> [sign extension] -> Rn
  class LdOp_Rmi_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPRMem:$Rm),
                         !strconcat(opcodestr, " @$Rm, $Rn")>;

  // (Rm) -> [sign extension] -> Rn, Rm+1 -> Rm
  class LdOp_Rminci_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPRMem:$Rm),
                         !strconcat(opcodestr, " @$Rm+, $Rn")>;

  // (disp + Rm) -> [sign extension] -> R0
  class LdOp_DispRm_R0<bits<16> op, string opcodestr> 
        : SHInstOP_M4_D4<op, (outs), (ins disp4:$disp, GPR:$Rm),
                         !strconcat(opcodestr, " @($disp, $Rm), R0")>;

  // (disp + Rm) -> [sign extension] -> Rn
  class LdOp_DispRm_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4_D4<op, (outs GPR:$Rn), (ins disp4:$disp, GPR:$Rm),
                            !strconcat(opcodestr, " @($disp, $Rm), $Rn")>;

  // (R0 + Rm) -> [sign extension] -> Rn
  class LdOp_RelR0Rm_Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPR:$Rm),
                         !strconcat(opcodestr, " @(R0, $Rm), $Rn")>;

  // (disp + GBR) -> [sign extension] -> R0
  class LdOp_DispGBR_R0<bits<16> op, string opcodestr> 
        : SHInstOP_D8<op, (outs), (ins disp8:$disp),
                         !strconcat(opcodestr, " @($disp, gbr),R0")>;
}

// STORE
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {


  // Rm -> [sign extension] -> (Rn)
  class StOp_Rm_Rni<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPR:$Rm),
                         !strconcat(opcodestr, " $Rm, @$Rn")>;

  // Rn-1 -> Rn, Rm -> [sign extension] -> (Rn)
  class StOp_Rm_Rndeci<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPRMem:$Rn), (ins GPR:$Rm),
                         !strconcat(opcodestr, " $Rm, @-$Rn")>;

  // R0 -> [sign extension] -> (disp + Rn)
  class StOp_R0_DispRn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_D4<op, (outs disp4:$disp, GPR:$Rn), (ins),
                         !strconcat(opcodestr, " R0, @( $disp, $Rn )")>;

  // Rm -> [sign extension] -> (disp + Rn)
  class StOp_Rm_DispRn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4_D4<op, (outs disp4:$disp, GPR:$Rn), (ins GPR:$Rm),
                            !strconcat(opcodestr, " $Rm, @( $disp, $Rn )")>;
  
  // Rm -> (R0 + Rn)
  class StOp_Rm_RelR0Rn<bits<16> op, string opcodestr> 
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins  GPR:$Rm),
                         !strconcat(opcodestr, " $Rm, @(R0, $Rn)")>;

  // R0 -> (disp + GBR)
  class StOp_R0_DispGBR<bits<16> op, string opcodestr> 
        : SHInstOP_D8<op, (outs disp8:$disp), (ins),
                         !strconcat(opcodestr, " R0, @($disp, gbr)")>;
}


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {

  // LOADS
  def MOV_Rm_Rn             : LdOp_Rm_Rn<0b0110000000000011, "mov">;
  def MOVA_dispPC_R0    : LdOp_DispPC_R0<0b1100011100000000, "mova">;
  def MOVW_dispPC_Rn    : LdOp_DispPC_Rn<0b1001000000000000, "mov.w">;
  def MOVL_dispPC_Rn    : LdOp_DispPC_Rn<0b1101000000000000, "mov.l">;
  def MOVB_Rmi_Rn          : LdOp_Rmi_Rn<0b0110000000000000, "mov.b">;
  def MOVW_Rmi_Rn          : LdOp_Rmi_Rn<0b0110000000000001, "mov.w">;
  def MOVL_Rmi_Rn          : LdOp_Rmi_Rn<0b0110000000000010, "mov.l">;
  def MOVB_Rminci_Rn    : LdOp_Rminci_Rn<0b0110000000000100, "mov.b">;
  def MOVW_Rminci_Rn    : LdOp_Rminci_Rn<0b0110000000000101, "mov.w">;
  def MOVL_Rminci_Rn    : LdOp_Rminci_Rn<0b0110000000000110, "mov.l">;
  def MOVB_dispRm_R0    : LdOp_DispRm_R0<0b1000010000000000, "mov.b">;
  def MOVW_dispRm_R0    : LdOp_DispRm_R0<0b1000010100000000, "mov.w">;
  def MOVL_dispRm_Rn    : LdOp_DispRm_Rn<0b0101000000000000, "mov.l">;
  def MOVB_RelR0Rm_Rn  : LdOp_RelR0Rm_Rn<0b0000000000001100, "mov.b">;
  def MOVW_RelR0Rm_Rn  : LdOp_RelR0Rm_Rn<0b0000000000001101, "mov.w">;
  def MOVL_RelR0Rm_Rn  : LdOp_RelR0Rm_Rn<0b0000000000001110, "mov.l">;
  def MOVB_dispGBR_R0  : LdOp_DispGBR_R0<0b1100010000000000, "mov.b">;
  def MOVW_dispGBR_R0  : LdOp_DispGBR_R0<0b1100010100000000, "mov.w">;
  def MOVL_dispGBR_R0  : LdOp_DispGBR_R0<0b1100011000000000, "mov.l">;

  def MOVRT_Rn                 : LdOp_Rn<0b0000000000111001, "movrt">;
  def MOVT_Rn                  : LdOp_Rn<0b0000000000101001, "movt">;

  // STORES
  def MOV_imm_Rn           : LdOp_Imm_Rn<0b1110000000000000, "mov">;
  def MOVB_Rm_Rni          : StOp_Rm_Rni<0b0010000000000000, "mov.b">;
  def MOVW_Rm_Rni          : StOp_Rm_Rni<0b0010000000000001, "mov.w">;
  def MOVL_Rm_Rni          : StOp_Rm_Rni<0b0010000000000010, "mov.l">;
  def MOVB_Rm_Rndeci    : StOp_Rm_Rndeci<0b0010000000000100, "mov.b">;
  def MOVW_Rm_Rndeci    : StOp_Rm_Rndeci<0b0010000000000101, "mov.w">;
  def MOVL_Rm_Rndeci    : StOp_Rm_Rndeci<0b0010000000000110, "mov.l">;
  def MOVB_R0_dispRn    : StOp_R0_DispRn<0b1000000000000000, "mov.b">;
  def MOVW_R0_dispRn    : StOp_R0_DispRn<0b1000000100000000, "mov.w">;
  def MOVL_Rm_dispRn    : StOp_Rm_DispRn<0b0001000000000000, "mov.l">;
  def MOVB_Rm_RelR0Rn  : StOp_Rm_RelR0Rn<0b0000000000000100, "mov.b">;
  def MOVW_Rm_RelR0Rn  : StOp_Rm_RelR0Rn<0b0000000000000101, "mov.w">;
  def MOVL_Rm_RelR0Rn  : StOp_Rm_RelR0Rn<0b0000000000000110, "mov.l">;
  def MOVB_R0_dispGBR  : StOp_R0_DispGBR<0b1100000000000000, "mov.b">;
  def MOVW_R0_dispGBR  : StOp_R0_DispGBR<0b1100000100000000, "mov.w">;
  def MOVL_R0_dispGBR  : StOp_R0_DispGBR<0b1100001000000000, "mov.l">;

  // OTHER
  def NOTT                  :         Op<0b0000000001101000, "nott">;
  def SWAPB_Rm_Rn           :   Op_Rm_Rn<0b0110000000001000, "swap.b">;
  def SWAPW_Rm_Rn           :   Op_Rm_Rn<0b0110000000001001, "swap.w">;
  def XTRCT_Rm_Rn           :   Op_Rm_Rn<0b0010000000001101, "xtrct">;
}

// LOAD ALIASES
def MOVA_Disp_R0           : InstAlias<"mova $disp, R0", (MOVA_dispPC_R0 disp8:$disp), 0>;
def MOVW_Disp_Rn           : InstAlias<"mov.w $disp, $Rn", (MOVW_dispPC_Rn GPR:$Rn, disp8:$disp), 0>;
def MOVL_Disp_Rn           : InstAlias<"mov.l $disp, $Rn", (MOVL_dispPC_Rn GPR:$Rn, disp8:$disp), 0>;