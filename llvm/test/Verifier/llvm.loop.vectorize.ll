; Test "llvm.loop.vectorize.enable" / "llvm.loop.vectorize.disable"
; single-operand validation.

; DEFINE: %{VERIFY} = llvm-as -disable-output %t 2>&1

define void @test() {
entry:
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}

;      BAD: Expecting only the metadata name

; Single-operand enable.
; RUN: cat %s > %t
; RUN: echo '!1 = !{!"llvm.loop.vectorize.enable"}' >> %t
; RUN: %{VERIFY}

; Single-operand disable.
; RUN: cat %s > %t
; RUN: echo '!1 = !{!"llvm.loop.vectorize.disable"}' >> %t
; RUN: %{VERIFY}

; Two-operand enable with boolean false (legacy form, now rejected).
; RUN: cat %s > %t
; RUN: echo '!1 = !{!"llvm.loop.vectorize.enable", i1 0}' >> %t
; RUN: not %{VERIFY} | FileCheck %s -check-prefix=BAD

; Two-operand enable with boolean true (legacy form, now rejected).
; RUN: cat %s > %t
; RUN: echo '!1 = !{!"llvm.loop.vectorize.enable", i1 1}' >> %t
; RUN: not %{VERIFY} | FileCheck %s -check-prefix=BAD

; Two-operand disable (rejected).
; RUN: cat %s > %t
; RUN: echo '!1 = !{!"llvm.loop.vectorize.disable", i1 0}' >> %t
; RUN: not %{VERIFY} | FileCheck %s -check-prefix=BAD
