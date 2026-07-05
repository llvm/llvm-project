; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s -check-prefix=LINUX

; Verify that the 5th and 6th parameters are coming from the correct location
; on the stack.
define i32 @f6(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6) nounwind readnone optsize {
entry:
; CHECK: movl    48(%rsp), %eax
; CHECK: addl    40(%rsp), %eax
; LINUX: leal    (%r8,%r9), %eax
  %add = add nsw i32 %p6, %p5
  ret i32 %add
}

define win64cc i32 @f7(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6) nounwind readnone optsize {
entry:
; CHECK: movl    48(%rsp), %eax
; CHECK: addl    40(%rsp), %eax
; LINUX: movl    48(%rsp), %eax
; LINUX: addl    40(%rsp), %eax
  %add = add nsw i32 %p6, %p5
  ret i32 %add
}

; Verify that even though we're compiling for Windows, parameters behave as
; on other platforms here (note the x86_64_sysvcc calling convention).
define x86_64_sysvcc i32 @f8(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6) nounwind readnone optsize {
entry:
; CHECK: leal    (%r8,%r9), %eax
; LINUX: leal    (%r8,%r9), %eax
  %add = add nsw i32 %p6, %p5
  ret i32 %add
}
