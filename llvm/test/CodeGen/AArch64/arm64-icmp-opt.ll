; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; Optimize (x > -1) to (x >= 0) etc.
; Optimize (cmp (add / sub), 0): eliminate the subs used to update flag
;   for comparison only
; rdar://10233472

define i32 @t1(i64 %a) {
; CHECK-LABEL: t1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    lsr x8, x0, #63
; CHECK-NEXT:    eor w0, w8, #0x1
; CHECK-NEXT:    ret
;
  %cmp = icmp sgt i64 %a, -1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
