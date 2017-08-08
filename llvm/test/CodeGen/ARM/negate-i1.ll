; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s

; PR30660 - https://llvm.org/bugs/show_bug.cgi?id=30660

define i32 @select_i32_neg1_or_0(i1 %a) {
; CHECK-LABEL: select_i32_neg1_or_0:
; CHECK-NEXT:  @ BB#0:
; CHECK-NEXT:    and r0, r0, #1
; CHECK-NEXT:    rsb r0, r0, #0
; CHECK-NEXT:    mov pc, lr
;
  %b = sext i1 %a to i32
  ret i32 %b
}

define i32 @select_i32_neg1_or_0_zeroext(i1 zeroext %a) {
; CHECK-LABEL: select_i32_neg1_or_0_zeroext:
; CHECK-NEXT:  @ BB#0:
; CHECK-NEXT:    rsb r0, r0, #0
; CHECK-NEXT:    mov pc, lr
;
  %b = sext i1 %a to i32
  ret i32 %b
}

