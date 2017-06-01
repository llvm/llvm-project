; RUN: llc < %s -mtriple=s390x-redhat-linux | FileCheck %s
;
; Triggers a path in SelectionDAG's UpdateChains where a node is
; deleted but we try to read it later (pr31710), invoking UB in
; release mode or hitting an assert if they're enabled.

; CHECK: btldata:
define void @btldata(i64* %u0, i32** %p0, i32** %p1, i32** %p3, i32** %p5, i32** %p7) {
entry:
  %x0 = load i32*, i32** %p0, align 8, !tbaa !0
  store i64 0, i64* %u0, align 8, !tbaa !4
  %x1 = load i32*, i32** %p1, align 8, !tbaa !0
  %x2 = load i32, i32* %x1, align 4, !tbaa !6
  %x2ext = sext i32 %x2 to i64
  store i32 %x2, i32* %x1, align 4, !tbaa !6
  %x3 = load i32*, i32** %p3, align 8, !tbaa !0
  %ptr = getelementptr inbounds i32, i32* %x3, i64 %x2ext
  %x4 = load i32, i32* %ptr, align 4, !tbaa !6
  %x4inc = add nsw i32 %x4, 1
  store i32 %x4inc, i32* %ptr, align 4, !tbaa !6
  store i64 undef, i64* %u0, align 8, !tbaa !4
  %x5 = load i32*, i32** %p5, align 8, !tbaa !0
  %x6 = load i32, i32* %x5, align 4, !tbaa !6
  store i32 %x6, i32* %x5, align 4, !tbaa !6
  %x7 = load i32*, i32** %p7, align 8, !tbaa !0
  %x8 = load i32, i32* %x7, align 4, !tbaa !6
  %x8inc = add nsw i32 %x8, 1
  store i32 %x8inc, i32* %x7, align 4, !tbaa !6
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
