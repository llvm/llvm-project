; RUN: llc < %s -regalloc=fast -optimize-regalloc=0 -no-integrated-as | FileCheck %s
; RUN: llc < %s -regalloc=basic -no-integrated-as      | FileCheck %s
; RUN: llc < %s -regalloc=greedy -no-integrated-as     | FileCheck %s

; The 1st, 2nd, 3rd and 5th registers must all be different.  The registers
; referenced in the 4th and 6th operands must not be the same as the 1st or 5th
; operand.
;
; CHECK: 1st=[[A1:%...]]
; CHECK-NOT: [[A1]]
; CHECK: 2nd=[[A2:%...]]
; CHECK-NOT: [[A1]]
; CHECK-NOT: [[A2]]
; CHECK: 3rd=[[A3:%...]]
; CHECK-NOT: [[A1]]
; CHECK-NOT: [[A2]]
; CHECK-NOT: [[A3]]
; CHECK: 5th=[[A5:%...]]
; CHECK-NOT: [[A1]]
; CHECK-NOT: [[A5]]
; CHECK: =4th

; The 6th operand is an 8-bit register, and it mustn't alias the 1st and 5th.
; CHECK: 1%e[[S1:.]]x
; CHECK: 5%e[[S5:.]]x
; CHECK-NOT: %[[S1]]
; CHECK-NOT: %[[S5]]

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%struct.foo = type { i32, i32, ptr }

define i32 @get(ptr %c, ptr %state) nounwind {
entry:
	%0 = getelementptr %struct.foo, ptr %c, i32 0, i32 0		; <ptr> [#uses=2]
	%1 = getelementptr %struct.foo, ptr %c, i32 0, i32 1		; <ptr> [#uses=2]
	%2 = getelementptr %struct.foo, ptr %c, i32 0, i32 2		; <ptr> [#uses=2]
	%3 = load i32, ptr %0, align 4		; <i32> [#uses=1]
	%4 = load i32, ptr %1, align 4		; <i32> [#uses=1]
	%5 = load i8, ptr %state, align 1		; <i8> [#uses=1]
	%asmtmp = tail call { i32, i32, i32, i32 } asm sideeffect "#1st=$0 $1 2nd=$1 $2 3rd=$2 $4 5th=$4 $3=4th 1$0 1%eXx 5$4 5%eXx 6th=$5", "=&r,=r,=r,=*m,=&q,=*imr,1,2,*m,5,~{dirflag},~{fpsr},~{flags},~{cx}"(ptr elementtype(ptr) %2, ptr elementtype(i8) %state, i32 %3, i32 %4, ptr elementtype(ptr) %2, i8 %5) nounwind		; <{ i32, i32, i32, i32 }> [#uses=3]
	%asmresult = extractvalue { i32, i32, i32, i32 } %asmtmp, 0		; <i32> [#uses=1]
	%asmresult1 = extractvalue { i32, i32, i32, i32 } %asmtmp, 1		; <i32> [#uses=1]
	store i32 %asmresult1, ptr %0
	%asmresult2 = extractvalue { i32, i32, i32, i32 } %asmtmp, 2		; <i32> [#uses=1]
	store i32 %asmresult2, ptr %1
	ret i32 %asmresult
}
