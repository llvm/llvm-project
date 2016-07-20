; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; Optimize (x > -1) to (x >= 0) etc.
; Optimize (cmp (add / sub), 0): eliminate the subs used to update flag
;   for comparison only
; rdar://10233472

define i32 @t1(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: movn
; CHECK: cmp  x0, #0
; CHECK: cset w0, ge
  %cmp = icmp sgt i64 %a, -1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
