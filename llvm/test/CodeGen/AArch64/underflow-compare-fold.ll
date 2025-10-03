; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

; GitHub issue #161036

define i64 @underflow_compare_fold(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold
; CHECK:      // %bb.0:
; CHECK-NEXT: subs x8, x0, x1
; CHECK-NEXT: csel x0, x0, x8, lo
; CHECK-NEXT: ret
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}
