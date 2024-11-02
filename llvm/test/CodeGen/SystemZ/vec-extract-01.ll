; Verify ReplaceExtractVectorEltOfLoadWithNarrowedLoad fixes
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a memory copy of a v2i32 (via the constant pool).
define void @f1(ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: llihf [[REG:%r[0-5]]], 1000000
; CHECK: oilf [[REG]], 99999
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  store <2 x i32> <i32 1000000, i32 99999>, ptr %dest
  ret void
}
