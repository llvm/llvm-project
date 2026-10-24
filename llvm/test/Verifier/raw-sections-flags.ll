; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Lowest non-zero bit.
; CHECK: llvm.raw.sections entry operand 2 must be zero (flags are reserved for future use)
; CHECK-NEXT: !{!"__mydata", i32 8, i32 1, !"data"}

; Higher bit.
; CHECK: llvm.raw.sections entry operand 2 must be zero (flags are reserved for future use)
; CHECK-NEXT: !{!"__mydata", i32 8, i32 8, !"data"}

; Wider integer type.
; CHECK: llvm.raw.sections entry operand 2 must be zero (flags are reserved for future use)
; CHECK-NEXT: !{!"__mydata", i32 8, i64 4294967296, !"data"}

; CHECK-NOT: llvm.raw.sections

!llvm.raw.sections = !{!0, !1, !2, !3, !4}

; Valid: zero flags.
!0 = !{!"__mydata", i32 8, i32 0, !"data"}
!1 = !{!"__mydata", i32 8, i64 0, !"data"}
!2 = !{!"__mydata", i32 8, i32 1, !"data"}
!3 = !{!"__mydata", i32 8, i32 8, !"data"}
!4 = !{!"__mydata", i32 8, i64 4294967296, !"data"}
