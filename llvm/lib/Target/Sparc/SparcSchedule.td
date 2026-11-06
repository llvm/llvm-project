//===-- SparcSchedule.td - Describe the Sparc Itineraries ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

def IIC_iu_or_fpu_instr : InstrItinClass;
def IIC_iu_instr : InstrItinClass;
def IIC_ld : InstrItinClass;
def IIC_fpu_ld : InstrItinClass;
def IIC_fpu_ldd : InstrItinClass;
def IIC_fpu_normal_instr : InstrItinClass;
def IIC_fpu_fast_instr : InstrItinClass;
def IIC_jmp : InstrItinClass;
def IIC_call : InstrItinClass;
def IIC_ldd : InstrItinClass;
def IIC_st : InstrItinClass;
def IIC_std : InstrItinClass;
def IIC_iu_smul : InstrItinClass;
def IIC_iu_umul : InstrItinClass;
def IIC_iu_div : InstrItinClass;
def IIC_ticc : InstrItinClass;
def IIC_atomic : InstrItinClass;
def IIC_fpu_muls : InstrItinClass;
def IIC_fpu_muld : InstrItinClass;
def IIC_fpu_divs : InstrItinClass;
def IIC_fpu_divd : InstrItinClass;
def IIC_fpu_sqrts : InstrItinClass;
def IIC_fpu_sqrtd : InstrItinClass;
def IIC_fpu_abs : InstrItinClass;
def IIC_fpu_movs : InstrItinClass;
def IIC_fpu_negs : InstrItinClass;
def IIC_smac_umac : InstrItinClass;
def IIC_fpu_stod : InstrItinClass;
def IIC_fpu_cmp : InstrItinClass;
def IIC_branch : InstrItinClass;
def IIC_fp_branch : InstrItinClass;
def IIC_fpu_st : InstrItinClass;
def IIC_fpu_std : InstrItinClass;
