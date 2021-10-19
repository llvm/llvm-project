; RUN: opt < %s -simplifycfg -S | FileCheck %s
;
; Ensure that the loop metadata is preserved when converting the
; conditional branch to an unconditional.

define void @commondest_loopid(i1 %T) {
; CHECK-LABEL: @commondest_loopid(
; CHECK: !llvm.loop !0
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"loopprop"}
entry:
        br label %loop

loop:
        br i1 %T, label %loop, label %loop, !llvm.loop !0
}

!0 = distinct !{!0, !1}
!1 = !{!"loopprop"}
