; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Zero alignment.
; CHECK: llvm.raw.sections entry operand 1 must be a power of two (alignment)
; CHECK-NEXT: !{!"__mydata", i32 0, i32 1, !"data"}

; Non-power-of-two alignment.
; CHECK: llvm.raw.sections entry operand 1 must be a power of two (alignment)
; CHECK-NEXT: !{!"__mydata", i32 12, i32 1, !"data"}

; CHECK-NOT: llvm.raw.sections

!llvm.raw.sections = !{!0, !1, !2, !3}

; Valid: alignment 1 and 8.
!0 = !{!"__mydata", i32 1, i32 1, !"data"}
!1 = !{!"__mydata", i32 8, i32 1, !"data"}
!2 = !{!"__mydata", i32 0, i32 1, !"data"}
!3 = !{!"__mydata", i32 12, i32 1, !"data"}
