; PR1219
; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @test(i1 %X) {
; CHECK-LABEL: test:
; CHECK-NOT: ret
; CHECK: movl $1, %eax
; CHECK: ret

  %hvar2 = zext i1 %X to i32
	%C = icmp sgt i32 %hvar2, -1
	br i1 %C, label %cond_true15, label %cond_true
cond_true15:
  ret i32 1
cond_true:
  ret i32 2
}
