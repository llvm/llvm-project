; Test that older bitcode carrying the two-operand form of
; "llvm.loop.distribute.enable" is auto-upgraded to the single-operand
; enable/disable pair on load.
;
; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

define void @enable_true() {
entry:
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}

define void @enable_false() {
entry:
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !2
exit:
  ret void
}

; i1 true  -> single-operand enable.
; i1 false -> disable.
; CHECK: !{!"llvm.loop.distribute.enable"}
; CHECK: !{!"llvm.loop.distribute.disable"}
; The old two-operand nodes must be gone from the module.
; CHECK-NOT: llvm.loop.distribute.enable", i1
; CHECK-NOT: llvm.loop.distribute.disable", i1

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.distribute.enable", i1 false}
