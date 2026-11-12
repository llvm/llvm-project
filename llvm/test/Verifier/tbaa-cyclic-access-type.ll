; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s
; RUN: llvm-as -disable-verify < %s | opt -passes=verify -S | FileCheck %s --check-prefix=STRIP

; CHECK: Cycle detected in access type parent chain
; CHECK-NEXT: store i32 0, ptr %p, align 4, !tbaa
; CHECK: Cycle detected in access type parent chain
; CHECK-NEXT: store i32 1, ptr %p, align 4, !tbaa
; CHECK: Cycle detected in access type parent chain
; CHECK-NEXT: store i32 2, ptr %p, align 4, !tbaa
; CHECK: Cycle detected in access type parent chain
; CHECK-NEXT: store i32 4, ptr %p, align 4, !tbaa
; CHECK-NOT: Cycle detected

; The bitcode reader strips invalid TBAA.
; STRIP-LABEL: define void @f(
; STRIP-NOT: !tbaa
define void @f(ptr %p) {
  store i32 0, ptr %p, align 4, !tbaa !3
  store i32 1, ptr %p, align 4, !tbaa !5
  store i32 2, ptr %p, align 4, !tbaa !8
  store i32 3, ptr %p, align 4, !tbaa !10
  store i32 4, ptr %p, align 4, !tbaa !13
  ret void
}

!0 = !{!"root"}

; Two new-format type nodes referring to each other.
!1 = distinct !{!2, i64 4, !"cycle1"}
!2 = distinct !{!1, i64 4, !"cycle2"}
!3 = !{!1, !1, i64 0, i64 4}

; A new-format type node that is its own parent.
!4 = distinct !{!4, i64 4, !"self"}
!5 = !{!4, !4, i64 0, i64 4}

; A new-format type node whose old-format parent refers back to it.
!6 = distinct !{!7, i64 4, !"mixed"}
!7 = distinct !{!"old", !6, i64 0}
!8 = !{!6, !6, i64 0, i64 4}

; An acyclic new-format access type.
!9 = !{!0, i64 4, !"int"}
!10 = !{!9, !9, i64 0, i64 4}

; A new-format type node whose parent chain leads into a cycle it is not part
; of.
!11 = distinct !{!12, i64 4, !"lasso1"}
!12 = distinct !{!11, i64 4, !"lasso2"}
!13 = !{!14, !14, i64 0, i64 4}
!14 = !{!11, i64 4, !"lasso_tail"}
