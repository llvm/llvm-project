; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; Check that we add nuw to multiplies by a constant where we can infer that the
; multiply does not have unsigned overflow.
declare i32 @get_int();

define void @foo() {
  %a = call i32 @get_int(), !range !0
  %b = mul i32 %a, 4
  ; CHECK: %b
  ; CHECK-NEXT: --> (4 * %a)<nuw>
  ret void
}

!0 = !{i32 0, i32 100}
