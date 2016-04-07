target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"
; RUN: llc < %s -march=ppc64 -mcpu=a2q | FileCheck %s

define void @s452(i32 %inp1) nounwind {
entry:
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %entry
  %conv.4 = sitofp i32 %inp1 to double
  %conv.5 = sitofp i32 %inp1 to double
  %mul.4.v.i0.1 = insertelement <2 x double> undef, double %conv.4, i32 0
  %mul.4.v.i0.2 = insertelement <2 x double> %mul.4.v.i0.1, double %conv.5, i32 1
  %mul.4 = fmul <2 x double> %mul.4.v.i0.2, undef
  %add7.4 = fadd <2 x double> undef, %mul.4
  store <2 x double> %add7.4, <2 x double>* undef, align 16
  br i1 undef, label %for.end, label %for.body4

for.end:                                          ; preds = %for.body4
  unreachable
; CHECK-LABEL: @s452
; CHECK: lfiwax [[REG1:[0-9]+]],
; CHECK: fcfid [[REG2:[0-9]+]], [[REG1]]
; FIXME: We could 'promote' this to a vector earlier and remove this splat.
; CHECK: qvesplati {{[0-9]+}}, [[REG2]], 0
; CHECK: qvfmul
; CHECK: qvfadd
; CHECK: qvesplati {{[0-9]+}},
; FIXME: We can use qvstfcdx here instead of two stores.
; CHECK: stfd
; CHECK: stfd
}

