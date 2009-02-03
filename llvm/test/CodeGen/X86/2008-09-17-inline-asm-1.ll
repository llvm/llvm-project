; RUN: llvm-as < %s | llc -march=x86 | not grep "movl %eax, %eax"
; RUN: llvm-as < %s | llc -march=x86 | not grep "movl %edx, %edx"
; RUN: llvm-as < %s | llc -march=x86 | not grep "movl (%eax), %eax"
; RUN: llvm-as < %s | llc -march=x86 | not grep "movl (%edx), %edx"
; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | not grep "movl %eax, %eax"
; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | not grep "movl %edx, %edx"
; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | not grep "movl (%eax), %eax"
; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | not grep "movl (%edx), %edx"

; %0 must not be put in EAX or EDX.
; In the first asm, $0 and $2 must not be put in EAX.
; In the second asm, $0 and $2 must not be put in EDX.
; This is kind of hard to test thoroughly, but the things above should continue
; to pass, I think.
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@x = common global i32 0		; <i32*> [#uses=1]

define i32 @aci(i32* %pw) nounwind {
entry:
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	%asmtmp = tail call { i32, i32 } asm "movl $0, %eax\0A\090:\0A\09test %eax, %eax\0A\09je 1f\0A\09movl %eax, $2\0A\09incl $2\0A\09lock\0A\09cmpxchgl $2, $0\0A\09jne 0b\0A\091:", "=*m,=&{ax},=&r,*m,~{dirflag},~{fpsr},~{flags},~{memory},~{cc}"(i32* %pw, i32* %pw) nounwind		; <{ i32, i32 }> [#uses=0]
	%asmtmp2 = tail call { i32, i32 } asm "movl $0, %edx\0A\090:\0A\09test %edx, %edx\0A\09je 1f\0A\09movl %edx, $2\0A\09incl $2\0A\09lock\0A\09cmpxchgl $2, $0\0A\09jne 0b\0A\091:", "=*m,=&{dx},=&r,*m,~{dirflag},~{fpsr},~{flags},~{memory},~{cc}"(i32* %pw, i32* %pw) nounwind		; <{ i32, i32 }> [#uses=1]
	%asmresult3 = extractvalue { i32, i32 } %asmtmp2, 0		; <i32> [#uses=1]
	%1 = add i32 %asmresult3, %0		; <i32> [#uses=1]
	ret i32 %1
}
