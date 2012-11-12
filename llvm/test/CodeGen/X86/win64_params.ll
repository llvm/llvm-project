; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s

; Verify that the 5th and 6th parameters are coming from the correct location
; on the stack.
define i32 @f6(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6) nounwind readnone optsize {
entry:
; CHECK: movl    48(%rsp), %eax
; CHECK: addl    40(%rsp), %eax
  %add = add nsw i32 %p6, %p5
  ret i32 %add
}
