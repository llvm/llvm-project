; Test that carry-less multiplication is expanded without the vector
; facility.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare i32 @llvm.clmul.i32(i32, i32)

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK-NOT: vgfm
; CHECK-NOT: brasl
; CHECK: br %r14
  %res = call i32 @llvm.clmul.i32(i32 %a, i32 %b)
  ret i32 %res
}
