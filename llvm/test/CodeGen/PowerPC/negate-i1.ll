; RUN: llc < %s -mtriple=powerpc64-apple-darwin | FileCheck %s

; PR30661 - https://llvm.org/bugs/show_bug.cgi?id=30661

define i32 @select_i32_neg1_or_0(i1 %a) {
; CHECK-LABEL: select_i32_neg1_or_0:
; CHECK:       ; BB#0:
; CHECK-NEXT:    sldi r2, r3, 63
; CHECK-NEXT:    sradi r3, r2, 63
; CHECK-NEXT:    blr
;
  %b = sext i1 %a to i32
  ret i32 %b
}

define i32 @select_i32_neg1_or_0_zeroext(i1 zeroext %a) {
; CHECK-LABEL: select_i32_neg1_or_0_zeroext:
; CHECK:       ; BB#0:
; CHECK-NEXT:    sldi r2, r3, 63
; CHECK-NEXT:    sradi r3, r2, 63
; CHECK-NEXT:    blr
;
  %b = sext i1 %a to i32
  ret i32 %b
}

