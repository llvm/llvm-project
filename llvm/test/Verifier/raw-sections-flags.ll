; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Write without Alloc.
; CHECK: llvm.raw.sections entry operand 2 is not a valid combination of section flags
; CHECK-NEXT: !{!"__mydata", i32 8, i32 2, !"data"}

; Exec without Alloc.
; CHECK: llvm.raw.sections entry operand 2 is not a valid combination of section flags
; CHECK-NEXT: !{!"__mydata", i32 8, i32 4, !"data"}

; Write combined with Exec.
; CHECK: llvm.raw.sections entry operand 2 is not a valid combination of section flags
; CHECK-NEXT: !{!"__mydata", i32 8, i32 7, !"data"}

; Exclude combined with Alloc.
; CHECK: llvm.raw.sections entry operand 2 is not a valid combination of section flags
; CHECK-NEXT: !{!"__mydata", i32 8, i32 9, !"data"}

; Unknown flag bit.
; CHECK: llvm.raw.sections entry operand 2 is not a valid combination of section flags
; CHECK-NEXT: !{!"__mydata", i32 8, i32 16, !"data"}

!llvm.raw.sections = !{!0, !1, !2, !3, !4, !5}

; Valid: Alloc only.
!0 = !{!"__mydata", i32 8, i32 1, !"data"}
!1 = !{!"__mydata", i32 8, i32 2, !"data"}
!2 = !{!"__mydata", i32 8, i32 4, !"data"}
!3 = !{!"__mydata", i32 8, i32 7, !"data"}
!4 = !{!"__mydata", i32 8, i32 9, !"data"}
!5 = !{!"__mydata", i32 8, i32 16, !"data"}
