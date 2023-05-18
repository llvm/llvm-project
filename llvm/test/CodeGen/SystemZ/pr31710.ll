; RUN: llc < %s -mtriple=s390x-redhat-linux | FileCheck %s
;
; Triggers a path in SelectionDAG's UpdateChains where a node is
; deleted but we try to read it later (pr31710), invoking UB in
; release mode or hitting an assert if they're enabled.

; CHECK: btldata:
define void @btldata(ptr %u0, ptr %p0, ptr %p1, ptr %p3, ptr %p5, ptr %p7) {
entry:
  %x0 = load ptr, ptr %p0, align 8, !tbaa !0
  store i64 0, ptr %u0, align 8, !tbaa !4
  %x1 = load ptr, ptr %p1, align 8, !tbaa !0
  %x2 = load i32, ptr %x1, align 4, !tbaa !6
  %x2ext = sext i32 %x2 to i64
  store i32 %x2, ptr %x1, align 4, !tbaa !6
  %x3 = load ptr, ptr %p3, align 8, !tbaa !0
  %ptr = getelementptr inbounds i32, ptr %x3, i64 %x2ext
  %x4 = load i32, ptr %ptr, align 4, !tbaa !6
  %x4inc = add nsw i32 %x4, 1
  store i32 %x4inc, ptr %ptr, align 4, !tbaa !6
  store i64 undef, ptr %u0, align 8, !tbaa !4
  %x5 = load ptr, ptr %p5, align 8, !tbaa !0
  %x6 = load i32, ptr %x5, align 4, !tbaa !6
  store i32 %x6, ptr %x5, align 4, !tbaa !6
  %x7 = load ptr, ptr %p7, align 8, !tbaa !0
  %x8 = load i32, ptr %x7, align 4, !tbaa !6
  %x8inc = add nsw i32 %x8, 1
  store i32 %x8inc, ptr %x7, align 4, !tbaa !6
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !2, i64 0}
