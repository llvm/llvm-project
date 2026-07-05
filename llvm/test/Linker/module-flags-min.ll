; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-link a.ll b.ll -S -o - 2>&1 | FileCheck %s

; CHECK:      !0 = !{i32 8, !"foo", i16 2}
; CHECK-NEXT: !1 = !{i32 8, !"bar", i64 3}
; CHECK-NEXT: !2 = !{i32 8, !"only_in_a", i32 0}
; CHECK-NEXT: !3 = !{i32 8, !"required_in_b", i32 3}
; CHECK-NEXT: !4 = !{i32 8, !"only_in_b", i32 0}
; CHECK-NEXT: !5 = !{i32 3, !"require", !6}
; CHECK-NEXT: !6 = !{!"required_in_b", i32 3}

;--- a.ll
!0 = !{ i32 8, !"foo", i16 2 }
!1 = !{ i32 8, !"bar", i64 4 }
!2 = !{ i32 8, !"only_in_a", i32 4 }
!3 = !{ i32 8, !"required_in_b", i32 3 }

!llvm.module.flags = !{ !0, !1, !2, !3 }

;--- b.ll
!0 = !{ i32 8, !"foo", i16 3 }
!1 = !{ i32 8, !"bar", i64 3 }
!2 = !{ i32 8, !"only_in_b", i32 3 }
!3 = !{ i32 8, !"required_in_b", i32 3 }
!4 = !{ i32 3, !"require", !{!"required_in_b", i32 3} }

!llvm.module.flags = !{ !0, !1, !2, !3, !4 }
