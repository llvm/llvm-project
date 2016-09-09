; RUN: llc -march=mipsel -mcpu=mips32 -relocation-model=pic < %s
; RUN: llc -march=mipsel -mcpu=mips32 -pre-RA-sched=source -relocation-model=pic < %s | FileCheck %s --check-prefix=SOURCE-SCHED
; RUN: llc -march=mipsel -mcpu=mips32r2 -relocation-model=pic < %s
; RUN: llc -march=mipsel -mcpu=mips32r2 -pre-RA-sched=source -relocation-model=pic < %s | FileCheck %s --check-prefix=SOURCE-SCHED

@gf0 = external global float
@gf1 = external global float
@gd0 = external global double
@gd1 = external global double

define float @select_cc_f32(float %a, float %b) nounwind {
entry:
; SOURCE-SCHED: lui
; SOURCE-SCHED: addiu
; SOURCE-SCHED: addu
; SOURCE-SCHED: lw
; SOURCE-SCHED: sw
; SOURCE-SCHED: lw
; SOURCE-SCHED: lui
; SOURCE-SCHED: sw
; SOURCE-SCHED: c.olt.s
; SOURCE-SCHED: lw
; SOURCE-SCHED: lwc1
; SOURCE-SCHED: mtc1
; SOURCE-SCHED: jr
  store float 0.000000e+00, float* @gf0, align 4
  store float 1.000000e+00, float* @gf1, align 4
  %cmp = fcmp olt float %a, %b
  %conv = zext i1 %cmp to i32
  %conv1 = sitofp i32 %conv to float
  ret float %conv1
}

define double @select_cc_f64(double %a, double %b) nounwind {
entry:
  store double 0.000000e+00, double* @gd0, align 8
  store double 1.000000e+00, double* @gd1, align 8
  %cmp = fcmp olt double %a, %b
  %conv = zext i1 %cmp to i32
  %conv1 = sitofp i32 %conv to double
  ret double %conv1
}

