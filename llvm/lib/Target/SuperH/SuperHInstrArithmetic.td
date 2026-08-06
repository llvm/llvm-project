//===-- SuperHInstrArithmetic.td - SuperH Arithmetic Instructions -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes all of the arithmetic instructions in the SuperH
/// ISA.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Arithmetic Instructions
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {

  // ADDITION
  def ADD_Rm_Rn          : Op_Rm_Rn<0b0011000000001100, "add">;
  def ADD_imm_Rn        : Op_Imm_Rn<0b0111000000000000, "add">;
  def ADDC_Rm_Rn         : Op_Rm_Rn<0b0011000000001110, "addc">;
  def ADDV_Rm_Rn         : Op_Rm_Rn<0b0011000000001111, "addv">;

  // SUBTRACTION
  def DT_Rn                 : Op_Rn<0b0100000000010000, "dt">;
  def NEG_Rm_Rn          : Op_Rm_Rn<0b0110000000001011, "neg">;
  def NEGC_Rm_Rn         : Op_Rm_Rn<0b0110000000001010, "negc">;
  def SUB_Rm_Rn          : Op_Rm_Rn<0b0011000000001000, "sub">;
  def SUBC_Rm_Rn         : Op_Rm_Rn<0b0011000000001010, "subc">;
  def SUBV_Rm_Rn         : Op_Rm_Rn<0b0011000000001011, "subv">;

  // MULTIPLICATION
  def MUL_Rm_Rn          : Op_Rm_Rn<0b0000000000000111, "mul.l">;
  def MULR_R0_Rn         : Op_Rm_Rn<0b0100000010000000, "mulr">;
  def MULSW_Rm_Rn        : Op_Rm_Rn<0b0010000000001111, "muls.w">;
  def MULUW_Rm_Rn        : Op_Rm_Rn<0b0010000000001110, "mulu.w">;
  def DMULS_Rm_Rn        : Op_Rm_Rn<0b0011000000001101, "dmuls.l">;
  def DMULU_Rm_Rn        : Op_Rm_Rn<0b0011000000000101, "dmulu.l">;

  // DIVISION
  def DIV0S_Rm_Rn        : Op_Rm_Rn<0b0010000000000111, "div0s">;
  def DIV0U                    : Op<0b0000000000011001, "div0u">;
  def DIV1_Rm_Rn         : Op_Rm_Rn<0b0011000000000100, "div1">;
  def DIVS_R0_Rn         : Op_R0_Rn<0b0100000010010100, "divs">;
  def DIVU_R0_Rn         : Op_R0_Rn<0b0100000010000100, "divu">;

  // COMPARISON
  def CMPEQ_Imm_R0      : Op_Imm_R0<0b1000100000000000, "cmp/eq">;
  def CMPEQ_Rm_Rn        : Op_Rm_Rn<0b0011000000000000, "cmp/eq">;
  def CMPHS_Rm_Rn        : Op_Rm_Rn<0b0011000000000010, "cmp/hs">;
  def CMPGE_Rm_Rn        : Op_Rm_Rn<0b0011000000000011, "cmp/ge">;
  def CMPHI_Rm_Rn        : Op_Rm_Rn<0b0011000000000110, "cmp/hi">;
  def CMPGT_Rn              : Op_Rn<0b0011000000000111, "cmp/gt">;
  def CMPPL_Rn              : Op_Rn<0b0100000000010101, "cmp/pl">;
  def CMPPZ_Rn              : Op_Rn<0b0100000000010001, "cmp/pz">;
  def CMPSTR_Rm_Rn       : Op_Rm_Rn<0b0010000000001100, "cmp/str">;

  // INTEGER EXTENSION
  def EXTSB_Rm_Rn        : Op_Rm_Rn<0b0110000000001110, "exts.b">;
  def EXTSW_Rm_Rn        : Op_Rm_Rn<0b0110000000001111, "exts.w">;
  def EXTUB_Rm_Rn        : Op_Rm_Rn<0b0110000000001100, "extu.b">;
  def EXTUW_Rm_Rn        : Op_Rm_Rn<0b0110000000001101, "extu.w">;

  // LOGIC
  def AND_Rm_Rn          : Op_Rm_Rn<0b0010000000001001, "and">;
  def AND_Imm_R0        : Op_Imm_R0<0b1100100100000000, "and">;
  def NOT_Rm_Rn          : Op_Rm_Rn<0b0110000000000111, "not">;
  def OR_Rm_Rn           : Op_Rm_Rn<0b0010000000001011, "or">;
  def OR_Imm_R0         : Op_Imm_R0<0b1100101100000000, "or">;
  def TST_Rm_Rn          : Op_Rm_Rn<0b0010000000001000, "tst">;
  def TST_Imm_R0        : Op_Imm_R0<0b1100100000000000, "tst">;
  def XOR_Rm_Rn          : Op_Rm_Rn<0b0010000000001010, "xor">;
  def XOR_Imm_R0        : Op_Imm_R0<0b1100101000000000, "xor">;
}