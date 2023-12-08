; RUN: opt %s -codegenprepare -mattr=+soft-float -S | FileCheck %s -check-prefix=CHECK -check-prefix=SOFTFP
; RUN: opt %s -codegenprepare -mattr=-soft-float -S | FileCheck %s -check-prefix=CHECK -check-prefix=HARDFP

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @foo
; CHECK:       entry:
; SOFTFP:      fcmp
; HARDFP-NOT:  fcmp
; CHECK:       body:
; SOFTFP-NOT:  fcmp
; HARDFP:      fcmp
define void @foo(float %a, float %b) {
entry:
  %c = fcmp oeq float %a, %b
  br label %head
head:
  %IND = phi i32 [ 0, %entry ], [ %IND.new, %body1 ]
  %CMP = icmp slt i32 %IND, 1250
  br i1 %CMP, label %body, label %tail
body:
  br i1 %c, label %body1, label %tail
body1:
  %IND.new = add i32 %IND, 1
  br label %head
tail:
  ret void
}
