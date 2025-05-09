; RUN: llc < %s -mtriple=sparcv9 | FileCheck %s
  
; CHECK-LABEL: test
; CHECK: mov %i3, %o1
; CHECK-NEXT: mov %i2, %o0
; CHECK-NEXT: call __ashlti3
; CHECK-NEXT: srl %i1, 0, %o2
define i128 @test(i128 %a, i128 %b) {
entry:
    %tmp = shl i128 %b, %a
    ret i128 %tmp
}
