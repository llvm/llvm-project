; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 | grep TAILCALL
; Check that lowered arguments on the stack do not overwrite each other.
; Move param %in1 to temp register (%eax).
; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 -x86-asm-syntax=att | grep {movl	40(%rsp), %eax}
; Add %in1 %p1 to another temporary register (%r9d).
; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 -x86-asm-syntax=att | grep {movl	%edi, %r9d}
; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 -x86-asm-syntax=att | grep {addl	32(%rsp), %r9d}
; Move result of addition to stack.
; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 -x86-asm-syntax=att | grep {movl	%r9d, 40(%rsp)}
; Move param %in2 to stack.
; RUN: llvm-as < %s | llc -tailcallopt -march=x86-64 -x86-asm-syntax=att | grep {movl	%eax, 32(%rsp)}

declare fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %a, i32 %b)

define fastcc i32 @tailcaller(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %in1, i32 %in2) {
entry:
        %tmp = add i32 %in1, %p1
        %retval = tail call fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %in2,i32 %tmp)
        ret i32 %retval
}

