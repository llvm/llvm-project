; RUN: llc < %s -mcpu=generic -mtriple=i386-apple-darwin -asm-verbose=0 | FileCheck %s
; PR3149
; Make sure the copy after inline asm is not coalesced away.

; CHECK:         ## InlineAsm End
; CHECK-NEXT: BB0_2:
; CHECK-NEXT:    {{movl	%esi, %eax|addl	%edi, %esi}}


@"\01LC" = internal constant [7 x i8] c"n0=%d\0A\00"		; <[7 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32 (i64, i64)* @umoddi3 to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @umoddi3(i64 %u, i64 %v) nounwind noinline {
entry:
	%0 = trunc i64 %v to i32		; <i32> [#uses=2]
	%1 = trunc i64 %u to i32		; <i32> [#uses=4]
	%2 = lshr i64 %u, 32		; <i64> [#uses=1]
	%3 = trunc i64 %2 to i32		; <i32> [#uses=2]
	%4 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([7 x i8]* @"\01LC", i32 0, i32 0), i32 %1) nounwind		; <i32> [#uses=0]
	%5 = icmp ult i32 %1, %0		; <i1> [#uses=1]
	br i1 %5, label %bb2, label %bb

bb:		; preds = %entry
	%6 = lshr i64 %v, 32		; <i64> [#uses=1]
	%7 = trunc i64 %6 to i32		; <i32> [#uses=1]
	%asmtmp = tail call { i32, i32 } asm "subl $5,$1\0A\09sbbl $3,$0", "=r,=&r,0,imr,1,imr,~{dirflag},~{fpsr},~{flags}"(i32 %3, i32 %7, i32 %1, i32 %0) nounwind		; <{ i32, i32 }> [#uses=2]
	%asmresult = extractvalue { i32, i32 } %asmtmp, 0		; <i32> [#uses=1]
	%asmresult1 = extractvalue { i32, i32 } %asmtmp, 1		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb, %entry
	%n1.0 = phi i32 [ %asmresult, %bb ], [ %3, %entry ]		; <i32> [#uses=1]
	%n0.0 = phi i32 [ %asmresult1, %bb ], [ %1, %entry ]		; <i32> [#uses=1]
	%8 = add i32 %n0.0, %n1.0		; <i32> [#uses=1]
	ret i32 %8
}

declare i32 @printf(i8*, ...) nounwind
