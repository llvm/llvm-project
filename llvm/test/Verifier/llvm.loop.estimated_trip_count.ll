; Test "llvm.loop.estimated_trip_count" validation

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

;      BAD-VALUE: Expected second operand to be an integer constant of type i32 or smaller
; BAD-VALUE-NEXT: !1 = !{!"llvm.loop.estimated_trip_count",

;      TOO-FEW: Expected two operands
; TOO-FEW-NEXT: !1 = !{!"llvm.loop.estimated_trip_count"}

;      TOO-MANY: Expected two operands
; TOO-MANY-NEXT: !1 = !{!"llvm.loop.estimated_trip_count", i32 5, i32 5}

; No value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count"}' >> %t
; RUN: not %{RUN} TOO-FEW

; i16 value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", i16 5}' >> %t
; RUN: %{RUN} GOOD

; i32 value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", i32 5}' >> %t
; RUN: %{RUN} GOOD

; i64 value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", i64 5}' >> %t
; RUN: not %{RUN} BAD-VALUE

; MDString value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", !"5"}' >> %t
; RUN: not %{RUN} BAD-VALUE

; MDNode value.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", !2}' >> %t
; RUN: echo '!2 = !{i32 5}' >> %t
; RUN: not %{RUN} BAD-VALUE

; Too many values.
; RUN: cp %s %t
; RUN: chmod u+w %t
; RUN: echo '!1 = !{!"llvm.loop.estimated_trip_count", i32 5, i32 5}' >> %t
; RUN: not %{RUN} TOO-MANY
