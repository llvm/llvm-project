; RUN: llvm-as < %s | llc -march=x86 | grep "#%ebp %eax %edx 8(%esi) %ebx (%edi)"
; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | grep "#%ecx %eax %edx 8(%edi) %ebx (%esi)"
; The 1st, 2nd, 3rd and 5th registers above must all be different.  The registers
; referenced in the 4th and 6th operands must not be the same as the 1st or 5th
; operand.  There are many combinations that work; this is what llc puts out now.
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%struct.foo = type { i32, i32, i8* }

define i32 @get(%struct.foo* %c, i8* %state) nounwind {
entry:
	%0 = getelementptr %struct.foo* %c, i32 0, i32 0		; <i32*> [#uses=2]
	%1 = getelementptr %struct.foo* %c, i32 0, i32 1		; <i32*> [#uses=2]
	%2 = getelementptr %struct.foo* %c, i32 0, i32 2		; <i8**> [#uses=2]
	%3 = load i32* %0, align 4		; <i32> [#uses=1]
	%4 = load i32* %1, align 4		; <i32> [#uses=1]
	%5 = load i8* %state, align 1		; <i8> [#uses=1]
	%asmtmp = tail call { i32, i32, i32, i32 } asm sideeffect "#$0 $1 $2 $3 $4 $5", "=&r,=r,=r,=*m,=&q,=*imr,1,2,*m,5,~{dirflag},~{fpsr},~{flags},~{cx}"(i8** %2, i8* %state, i32 %3, i32 %4, i8** %2, i8 %5) nounwind		; <{ i32, i32, i32, i32 }> [#uses=3]
	%asmresult = extractvalue { i32, i32, i32, i32 } %asmtmp, 0		; <i32> [#uses=1]
	%asmresult1 = extractvalue { i32, i32, i32, i32 } %asmtmp, 1		; <i32> [#uses=1]
	store i32 %asmresult1, i32* %0
	%asmresult2 = extractvalue { i32, i32, i32, i32 } %asmtmp, 2		; <i32> [#uses=1]
	store i32 %asmresult2, i32* %1
	ret i32 %asmresult
}
