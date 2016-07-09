//===-- LeonFeatures.td - Describe the Leon Features -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CASA Support differs between LEON3-FT GR712RC and LEON3-FT UT699
// We need to have the option to switch this on and off.
//===----------------------------------------------------------------------===//

// support to casa instruction; for leon3 subtarget only
def LeonCASA : SubtargetFeature<
                   "hasleoncasa", "HasLeonCasa", "true",
                   "Enable CASA instruction for LEON3 and LEON4 processors">;

//===----------------------------------------------------------------------===//
// UMAC and SMAC support for LEON3 and LEON4 processors.
//===----------------------------------------------------------------------===//

// support to casa instruction; for leon3 subtarget only
def UMACSMACSupport
    : SubtargetFeature<"hasumacsmac", "HasUmacSmac", "true",
                       "Enable UMAC and SMAC for LEON3 and LEON4 processors">;

//===----------------------------------------------------------------------===//
// LEON Erratum fixes
//===----------------------------------------------------------------------===//

def ReplaceSDIV
    : SubtargetFeature<
          "replacesdiv", "PerformSDIVReplace", "true",
          "AT697E erratum fix: Do not emit SDIV, emit SDIVCC instead">;

def FixCALL
    : SubtargetFeature<"fixcall", "FixCallImmediates", "true",
                       "AT697E erratum fix: Restrict the size of the immediate "
                       "operand of the CALL instruction to 20 bits">;

def IgnoreZeroFlag
    : SubtargetFeature<"ignrzeroflag", "IgnoreZeroFlag", "true",
                       "AT697E erratum fix: Do not rely on the zero bit flag "
                       "on a divide overflow for SDIVCC and UDIVCC">;

def InsertNOPDoublePrecision
    : SubtargetFeature<"insrtnopdblprcsn", "InsertNOPDoublePrecision", "true",
                       "LEON2 erratum fix: Insert a NOP before the double "
                       "precision floating point instruction">;

def FixFSMULD : SubtargetFeature<"fixfsmuld", "FixFSMULD", "true",
                                 "LEON3 erratum fix: Do not select FSMULD">;

def ReplaceFMULS
    : SubtargetFeature<"replacefmuls", "ReplaceFMULS", "true",
                       "LEON3 erratum fix: Replace FMULS instruction with a "
                       "routine using conversions/double precision operations "
                       "to replace FMULS">;

def PreventRoundChange
    : SubtargetFeature<"prvntroundchange", "PreventRoundChange", "true",
                       "LEON3 erratum fix: Prevent any rounding mode change "
                       "request: use only the round-to-nearest rounding mode">;

def FixAllFDIVSQRT
    : SubtargetFeature<"fixallfdivsqrt", "FixAllFDIVSQRT", "true",
                       "LEON3 erratum fix: Fix FDIVS/FDIVD/FSQRTS/FSQRTD "
                       "instructions with NOPs and floating-point store">;

def InsertNOPLoad
    : SubtargetFeature<"insertnopload", "InsertNOPLoad", "true",
                       "LEON3 erratum fix: Insert a NOP instruction after "
                       "every single-cycle load instruction when the next "
                       "instruction is another load/store instruction">;

def FlushCacheLineSWAP
    : SubtargetFeature<"flshcachelineswap", "FlushCacheLineSWAP", "true",
                       "LEON3 erratum fix: Flush cache line containing the "
                       "lock before performing any of the atomic instructions "
                       "SWAP and LDSTUB">;

def InsertNOPsLoadStore
    : SubtargetFeature<"insertnopsloadstore", "InsertNOPsLoadStore", "true",
                       "LEON3 erratum fix: Insert NOPs between "
                       "single-precision loads and the store, so the number of "
                       "instructions between is 4">;
