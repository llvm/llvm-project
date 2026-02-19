; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O2 %s -o - | FileCheck %s

declare i32 @llvm.umin.i32(i32, i32)

define i32 @f(i1 %c0, i1 %c1, i32 %x) {
; CHECK-LABEL: f:
; CHECK: bic w0, w{{[0-9]+}}, w2
; CHECK-NOT: add w{{[0-9]+}}, w2, w{{[0-9]+}}
; CHECK-NOT: and w0, w{{[0-9]+}}, w{{[0-9]+}}
; CHECK: ret
entry:
  %a = select i1 %c0, i32 4, i32 0
  %e = zext i1 %c1 to i32
  %b = shl i32 1, %e
  %y = call i32 @llvm.umin.i32(i32 %a, i32 %b)
  %sum = add i32 %x, %y
  %r = and i32 %y, %sum
  ret i32 %r
}
