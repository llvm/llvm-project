; RUN: llc < %s -tailcallopt -mcpu=generic -mtriple=x86_64-linux -post-RA-scheduler=true | FileCheck %s
; RUN: llc < %s -tailcallopt -mcpu=generic -mtriple=x86_64-win32 -post-RA-scheduler=true | FileCheck %s

; FIXME: Redundant unused stack allocation could be eliminated.
; CHECK: subq  ${{24|72|80}}, %rsp

; Check that lowered arguments on the stack do not overwrite each other.
; Add %in1 %p1 to a different temporary register (%eax).
; CHECK: movl  [[A1:32|144]](%rsp), [[R1:%e..]]
; Move param %in1 to temp register (%r10d).
; CHECK: movl  [[A2:40|152]](%rsp), [[R2:%[a-z0-9]+]]
; Add %in1 %p1 to a different temporary register (%eax).
; CHECK: addl {{%edi|%ecx}}, [[R1]]
; Move param %in2 to stack.
; CHECK: movl  [[R2]], [[A1]](%rsp)
; Move result of addition to stack.
; CHECK: movl  [[R1]], [[A2]](%rsp)
; Eventually, do a TAILCALL
; CHECK: TAILCALL

declare fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %a, i32 %b) nounwind

define fastcc i32 @tailcaller(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %in1, i32 %in2) nounwind {
entry:
        %tmp = add i32 %in1, %p1
        %retval = tail call fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %in2,i32 %tmp)
        ret i32 %retval
}
