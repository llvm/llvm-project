; RUN: llc < %s -march=arm -mcpu=cortex-a8 2>&1 | FileCheck %s

; Check for error message:
; CHECK: non-trivial scalar-to-vector conversion, possible invalid constraint for vector type

define void @f() nounwind ssp {
  %1 = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } asm "vldm $4, { ${0:q}, ${1:q}, ${2:q}, ${3:q} }", "=r,=r,=r,=r,r"(i64* undef) nounwind, !srcloc !0
  ret void
}

!0 = metadata !{i32 318437}
