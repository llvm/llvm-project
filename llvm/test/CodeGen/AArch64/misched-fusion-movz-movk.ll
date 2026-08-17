; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=+fuse-movz-movk,+use-postra-scheduler | FileCheck %s --check-prefixes=CHECK

;; Lack post-RA scheduling, but should behave the same by plain pseudo-expansion
; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-a13 -mattr=+fuse-movz-movk | FileCheck %s --check-prefixes=CHECK
; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-a14 | FileCheck %s --check-prefixes=CHECK
; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-m5 | FileCheck %s --check-prefixes=CHECK

; CHECK-LABEL: movz_movk16_w:
; CHECK:      mov  [[R:w[0-9]+]], #48879
; CHECK-NEXT: movk [[R]], #61536, lsl #16
define i32 @movz_movk16_w(i32 %a, i32 %b) {
  %c = add i32 %a, -262095121
  %r = add i32 %c, %b
  ret i32 %r
}

; CHECK-LABEL: movz_movk32_x:
; CHECK:      mov  [[R:x[0-9]+]], #4660
; CHECK-NEXT: movk [[R]], #22136, lsl #32
define i64 @movz_movk32_x(i64 %a, i64 %b) {
  %c = add i64 %a, 95073396068916
  %r = add i64 %c, %b
  ret i64 %r
}

; CHECK-LABEL: movz_movk48_x:
; CHECK:      mov  [[R:x[0-9]+]], #4660
; CHECK-NEXT: movk [[R]], #22136, lsl #48
define i64 @movz_movk48_x(i64 %a, i64 %b) {
  %c = add i64 %a, 6230730084467085876
  %r = add i64 %c, %b
  ret i64 %r
}

; CHECK-LABEL: movz16_movk32_x:
; CHECK:      mov  [[R:x[0-9]+]], #305397760
; CHECK-NEXT: movk [[R]], #22136, lsl #32
define i64 @movz16_movk32_x(i64 %a, i64 %b) {
  %c = add i64 %a, 95073701462016
  %r = add i64 %c, %b
  ret i64 %r
}

; CHECK-LABEL: movz16_movk48_x:
; CHECK:      mov  [[R:x[0-9]+]], #305397760
; CHECK-NEXT: movk [[R]], #22136, lsl #48
define i64 @movz16_movk48_x(i64 %a, i64 %b) {
  %c = add i64 %a, 6230730084772478976
  %r = add i64 %c, %b
  ret i64 %r
}

; CHECK-LABEL: movz32_movk48_x:
; CHECK:      mov  [[R:x[0-9]+]], #20014547599360
; CHECK-NEXT: movk [[R]], #22136, lsl #48
define i64 @movz32_movk48_x(i64 %a, i64 %b) {
  %c = add i64 %a, 6230750099014680576
  %r = add i64 %c, %b
  ret i64 %r
}
