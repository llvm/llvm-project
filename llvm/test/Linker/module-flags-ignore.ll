; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-link a.ll b.ll -S -o - 2>&1 | FileCheck %s

; Test the 'ignore' behavior (9).

; CHECK: !llvm.module.flags = !{!0, !1, !2}
; CHECK: !0 = !{i32 9, !"foo", i32 1}
; CHECK: !1 = !{i32 9, !"bar", i32 4}
; CHECK: !2 = !{i32 9, !"only_in_b", i32 3}

;--- a.ll
!0 = !{ i32 9, !"foo", i32 1 }
!1 = !{ i32 9, !"bar", i32 4 }

!llvm.module.flags = !{ !0, !1 }

;--- b.ll
!0 = !{ i32 9, !"foo", i32 2 }
!1 = !{ i32 9, !"only_in_b", i32 3 }

!llvm.module.flags = !{ !0, !1 }
