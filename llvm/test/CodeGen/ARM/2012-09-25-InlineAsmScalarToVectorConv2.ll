; RUN: llc < %s -march=arm -mcpu=cortex-a8 2>&1 | FileCheck %s

; Check for error message:
; CHECK: scalar-to-vector conversion failed, possible invalid constraint for vector type

define hidden void @f(i32* %corr, i32 %order) nounwind ssp {
  tail call void asm sideeffect "vst1.s32 { ${1:q}, ${2:q} }, [$0]", "r,{q0},{q1}"(i32* %corr, <2 x i64>* undef, <2 x i64>* undef) nounwind, !srcloc !0
  ret void
}

!0 = metadata !{i32 257}
