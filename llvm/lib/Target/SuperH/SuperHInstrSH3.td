//===-- SuperHInstrSH3.td - DSP SuperH Instruction Definition --*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
///
/// \file
/// This file describes the SH3 extensions for the SuperH Instruction Set.
///
//===-------------------------------------------------------------------------===//

let Predicates = [IsSH3] in {




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Shift Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, Constraints = "$src = $Rn" in {

  // shad Rm,Rn
  let Defs = [SR] in
  def SHAD          : InstRmRn<0b0100000000001100, 
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "shad $Rm,$Rn",
                              []>;

  // shld Rm,Rn
  let Defs = [SR] in
  def SHLD          : InstRmRn<0b0100000000001101, 
                              (outs GPR:$Rn), (ins GPR:$Rm, GPR:$src),
                              "shld $Rm,$Rn",
                              []>;

}




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          System Control Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// clrs
let Defs = [SR] in
def CLRS                : Inst<0b0000000001001000, (outs), (ins), "clrs", []>;

// ldc Rm, SSR
let Defs = [SSR], isDelayIllegal = 1 in
def LDCSSR            : InstRm<0b0100000000111110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,ssr",
                              []>;

// ldc.l @Rm+, SSR
let Defs = [SSR], isDelayIllegal = 1 in
def LDCMSSR           : InstRm<0b0100000000110111, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,ssr",
                              []>;
// ldc Rm, SPC
let Defs = [SPC], isDelayIllegal = 1 in
def LDCSPC            : InstRm<0b0100000001001110, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,spc",
                              []>;

// ldc.l @Rm+, SPC
let Defs = [SPC], isDelayIllegal = 1 in
def LDCMSPC           : InstRm<0b0100000001000111, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,spc",
                              []>;

// TODO: LDCRn_BANK
// TODO: LDCMRn_BANK

// ldtlb
def LDTLB               : Inst<0b0000000000111000, (outs), (ins), "ldtlb", []>;

// pref @Rn
def PREF              : InstRn<0b0000000010000011, 
                              (outs), (ins IGPR:$Rn), 
                              "pref $Rn",
                              []>;

// sets
let Defs = [SR] in
def SETS                : Inst<0b0000000001011000, (outs), (ins), "sets", []>;

// stc SSR,Rn
let Uses = [SSR], isDelayIllegal = 1 in
def STCSSR            : InstRn<0b0000000000110010, 
                              (outs GPR:$Rn), (ins), 
                              "stc ssr,$Rn",
                              []>;

// stc.l SSR,@-Rn
let Uses = [SSR], isDelayIllegal = 1 in
def STCMSSR           : InstRn<0b0100000000110011, 
                              (outs), (ins IGPRDec:$Rn), 
                              "stc.l ssr,$Rn",
                              []>;

// stc SPC,Rn
let Uses = [SPC], isDelayIllegal = 1 in
def STCSPC            : InstRn<0b0000000001000010, 
                              (outs GPR:$Rn), (ins), 
                              "stc spc,$Rn",
                              []>;

// stc.l SPC,@-Rn
let Uses = [SPC], isDelayIllegal = 1 in
def STCMSPC           : InstRn<0b0100000001000011, 
                              (outs), (ins IGPRDec:$Rn), 
                              "stc.l spc,$Rn",
                              []>;

// TODO: STCRm_BANK
// TODO: STCMRm_BANK

}