; rdar://8465407
; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s

%struct.buf = type opaque

declare void @bar() nounwind optsize

define void @foo() nounwind optsize "no-frame-pointer-elim"="true" {
; CHECK-LABEL: foo:
; CHECK: push
; CHECK: add r7, sp
; CHECK: sub sp, #4
entry:
  %m.i = alloca %struct.buf*, align 4
  br label %bb

bb:
  br i1 undef, label %bb3, label %bb2

bb2:
  call void @bar() nounwind optsize
  br i1 undef, label %bb, label %bb3

bb3:
  br i1 undef, label %return, label %bb

return:
; CHECK: %bb3
; 'mov sp, r7' would have left sp in an invalid state
; CHECK-NOT: mov sp, r7
; CHECK-NOT: sub, sp, #4
; CHECK: add sp, #4
  ret void
}
