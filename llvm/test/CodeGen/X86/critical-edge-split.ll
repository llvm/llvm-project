; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -stats -info-output-file - | grep asm-printer | grep 31

	%CC = type { %Register }
	%II = type { %"struct.XX::II::$_74" }
	%JITFunction = type %YYValue* (%CC*, %YYValue**)
	%YYValue = type { i32 (...)** }
	%Register = type { %"struct.XX::ByteCodeFeatures" }
	%"struct.XX::ByteCodeFeatures" = type { i32 }
	%"struct.XX::II::$_74" = type { i8* }
@llvm.used = appending global [1 x i8*] [ i8* bitcast (%JITFunction* @loop to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define %YYValue* @loop(%CC*, %YYValue**) nounwind {
; <label>:2
	%3 = getelementptr %CC* %0, i32 -9		; <%CC*> [#uses=1]
	%4 = bitcast %CC* %3 to %YYValue**		; <%YYValue**> [#uses=2]
	%5 = load %YYValue** %4		; <%YYValue*> [#uses=3]
	%unique_1.i = ptrtoint %YYValue* %5 to i1		; <i1> [#uses=1]
	br i1 %unique_1.i, label %loop, label %11

loop:		; preds = %6, %2
	%.1 = phi %YYValue* [ inttoptr (i32 1 to %YYValue*), %2 ], [ %intAddValue, %6 ]		; <%YYValue*> [#uses=3]
	%immediateCmp = icmp slt %YYValue* %.1, %5		; <i1> [#uses=1]
	br i1 %immediateCmp, label %6, label %8

; <label>:6		; preds = %loop
	%lhsInt = ptrtoint %YYValue* %.1 to i32		; <i32> [#uses=1]
	%7 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %lhsInt, i32 2)		; <{ i32, i1 }> [#uses=2]
	%intAdd = extractvalue { i32, i1 } %7, 0		; <i32> [#uses=1]
	%intAddValue = inttoptr i32 %intAdd to %YYValue*		; <%YYValue*> [#uses=1]
	%intAddOverflow = extractvalue { i32, i1 } %7, 1		; <i1> [#uses=1]
	br i1 %intAddOverflow, label %.loopexit, label %loop

; <label>:8		; preds = %loop
	ret %YYValue* inttoptr (i32 10 to %YYValue*)

.loopexit:		; preds = %6
	%9 = bitcast %CC* %0 to %YYValue**		; <%YYValue**> [#uses=1]
	store %YYValue* %.1, %YYValue** %9
	store %YYValue* %5, %YYValue** %4
	%10 = call fastcc %YYValue* @foobar(%II* inttoptr (i32 3431104 to %II*), %CC* %0, %YYValue** %1)		; <%YYValue*> [#uses=1]
	ret %YYValue* %10

; <label>:11		; preds = %2
	%12 = call fastcc %YYValue* @foobar(%II* inttoptr (i32 3431080 to %II*), %CC* %0, %YYValue** %1)		; <%YYValue*> [#uses=1]
	ret %YYValue* %12
}

declare fastcc %YYValue* @foobar(%II*, %CC*, %YYValue**) nounwind

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind
