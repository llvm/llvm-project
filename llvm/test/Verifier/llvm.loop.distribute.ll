; Test "llvm.loop.distribute.enable" / "llvm.loop.distribute.disable"
; single-operand validation.

; DEFINE: %{RUN} = opt -passes=verify %t -disable-output 2>&1 | \
; DEFINE:   FileCheck %s -allow-empty -check-prefix

define void @test() {
entry:
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}

; GOOD-NOT: {{.}}

;      BAD: Expected one operand for llvm.loop.distribute metadata

; Single-operand enable.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.distribute.enable"}' >> %t
; RUN: %{RUN} GOOD

; Single-operand disable.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.distribute.disable"}' >> %t
; RUN: %{RUN} GOOD

; Two-operand enable with boolean false (legacy form, now rejected).
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.distribute.enable", i1 0}' >> %t
; RUN: not %{RUN} BAD

; Two-operand enable with boolean true (legacy form, now rejected).
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.distribute.enable", i1 1}' >> %t
; RUN: not %{RUN} BAD

; Two-operand disable (rejected).
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.distribute.disable", i1 0}' >> %t
; RUN: not %{RUN} BAD
