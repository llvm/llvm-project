//===-- AArch64GlobalISelPatterns.td - GlobalISel patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Selection and combine patterns for GlobalISel.
//
//===----------------------------------------------------------------------===//


//unpredicate patterns



// fneg
def : Pat<(nxv2f64 (fneg nxv2f64:$src)),
          (FNEG_ZPmZ_D_UNDEF (IMPLICIT_DEF), (PTRUE_D 31), ZPR:$src)>;

def : Pat<(nxv4f32 (fneg nxv4f32:$src)),
          (FNEG_ZPmZ_S_UNDEF (IMPLICIT_DEF), (PTRUE_S 31), ZPR:$src)>;

def : Pat<(nxv8f16 (fneg nxv8f16:$src)),
          (FNEG_ZPmZ_H_UNDEF (IMPLICIT_DEF), (PTRUE_H 31), ZPR:$src)>;

// fabs
def : Pat<(nxv2f64 (fabs nxv2f64:$src)),
          (FABS_ZPmZ_D_UNDEF (IMPLICIT_DEF), (PTRUE_D 31), ZPR:$src)>;

def : Pat<(nxv4f32 (fabs nxv4f32:$src)),
          (FABS_ZPmZ_S_UNDEF (IMPLICIT_DEF), (PTRUE_S 31), ZPR:$src)>;

def : Pat<(nxv8f16 (fabs nxv8f16:$src)),
          (FABS_ZPmZ_H_UNDEF (IMPLICIT_DEF), (PTRUE_H 31), ZPR:$src)>;
