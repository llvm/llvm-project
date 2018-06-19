; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; Check that we convert
;   zext((a * b)<nuw>)
; to
;   (zext(a) * zext(b))<nuw>

declare i32 @get_int();

; Transform doesn't apply here, because %a lacks range metadata.
; CHECK-LABEL: @no_range
define void @no_range() {
  %a = call i32 @get_int()
  %b = mul i32 %a, 4
  %c = zext i32 %b to i64
  ; CHECK: %c
  ; CHECK-NEXT: --> (zext i32 (4 * %a) to i64)
  ret void
}

; CHECK-LABEL: @range
;
; This had to be disabled when r334428 was reverted.  We should enable this test
; when r334428 is reapplied with a fix.
define void @range() {
  %a = call i32 @get_int(), !range !0
  %b = mul i32 %a, 4
  %c = zext i32 %b to i64
  ; CHECK: %c
  ; CHECK-NEXT: --> (zext i32 (4 * %a) to i64)
  ret void
}

!0 = !{i32 0, i32 100}
