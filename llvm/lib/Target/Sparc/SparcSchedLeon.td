//===-- SparcSchedLeon.td - LEON scheduling ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the itineraries of the LEON processors.
//===----------------------------------------------------------------------===//


def LEONIU : FuncUnit; // integer unit
def LEONFPU : FuncUnit; // floating-point unit

// Ref: http://www.atmel.com/Images/doc4226.pdf

def LEON2Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [5, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [5, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [21, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [20, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [36, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [37, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [65, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>
]>;

def LEON3Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [4, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_smac_umac, [InstrStage<1, [LEONIU]>], [2, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [17, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [24, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [25, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>
]>;

def LEON4Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [4, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_smac_umac, [InstrStage<1, [LEONIU]>], [2, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [17, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [24, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [25, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>
]>;
