; RUN: llc < %s -mtriple=i686-linux -show-mc-encoding | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -show-mc-encoding | FileCheck %s

declare void @foo()
declare void @bar()

define void @f(i32 %x, i32 %y) optsize {
entry:
	%p = icmp eq i32 %x, %y
  br i1 %p, label %bb1, label %bb2
bb1:
  tail call void @foo()
  ret void
bb2:
  tail call void @bar()
  ret void
}

; CHECK-LABEL: f:
; CHECK: cmp
; CHECK: jne bar
; Check that the asm doesn't just look good, but uses the correct encoding.
; CHECK: encoding: [0x75,A]

; CHECK: jmp foo
