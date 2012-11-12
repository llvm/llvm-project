; RUN: llc < %s -O0 -verify-machineinstrs -regalloc=fast
; Previously we'd crash as out of registers on this input by clobbering all of
; the aliases.
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10.0.0"

define void @_Z8TestCasev() nounwind ssp {
entry:
  %a = alloca float, align 4
  %tmp = load float* %a, align 4
  call void asm sideeffect "", "w,~{s0},~{s16}"(float %tmp) nounwind, !srcloc !0
  ret void
}

!0 = metadata !{i32 109}
