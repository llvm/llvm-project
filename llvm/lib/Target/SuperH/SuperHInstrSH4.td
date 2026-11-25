//===-- SuperHInstrSH4.td - DSP SuperH Instruction Definition --*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
///
/// \file
/// This file describes the SH4 extensions for the SuperH Instruction Set.
///
//===-------------------------------------------------------------------------===//

let Predicates = [IsSH4] in {




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          System Control Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// ldc Rm, DBR
let Defs = [DBR], isDelayIllegal = 1 in
def LDCDBR            : InstRm<0b0100000011111010, 
                              (outs), (ins GPR:$Rm), 
                              "ldc $Rm,dbr",
                              []>;

// ldc.l @Rm+, DBR
let Defs = [DBR], isDelayIllegal = 1 in
def LDCMDBR           : InstRm<0b0100000011110110, 
                              (outs), (ins IGPRInc:$Rm), 
                              "ldc.l $Rm,dbr",
                              []>;

// movca.l R0,@Rn
let Uses = [R0], mayStore = 1 in
def MOVCAL            : InstRn<0b0000000011000011, 
                              (outs), (ins IGPR:$Rn), 
                              "movca.l r0,$Rn",
                              []>;

// ocbi @Rn
def OCBI              : InstRn<0b0000000010010011, 
                              (outs), (ins IGPR:$Rn), 
                              "ocbi $Rn",
                              []>;
// ocbp @Rn
def OCBP              : InstRn<0b0000000010100011, 
                              (outs), (ins IGPR:$Rn), 
                              "ocbp $Rn",
                              []>;
// ocbwb @Rn
def OCBWB             : InstRn<0b0000000010110011, 
                              (outs), (ins IGPR:$Rn), 
                              "ocbwb $Rn",
                              []>;

// stc SGR,Rn
let Uses = [SGR], isDelayIllegal = 1 in
def STCSGR            : InstRn<0b0000000000111010, 
                              (outs GPR:$Rn), (ins), 
                              "stc sgr,$Rn",
                              []>;

// stc.l SGR,@-Rn
let Uses = [SGR], isDelayIllegal = 1 in
def STCMSGR           : InstRn<0b0100000000110010, 
                              (outs), (ins IGPRDec:$Rn), 
                              "stc.l sgr,$Rn",
                              []>;

// stc SGR,Rn
let Uses = [DBR], isDelayIllegal = 1 in
def STCDBR            : InstRn<0b0000000011111010, 
                              (outs GPR:$Rn), (ins), 
                              "stc dbr,$Rn",
                              []>;

// stc.l SGR,@-Rn
let Uses = [DBR], isDelayIllegal = 1 in
def STCMDBR           : InstRn<0b0100000011110010, 
                              (outs), (ins IGPRDec:$Rn), 
                              "stc.l dbr,$Rn",
                              []>;
}