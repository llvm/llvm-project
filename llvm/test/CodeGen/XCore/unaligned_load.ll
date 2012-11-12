; RUN: llc < %s -march=xcore > %t1.s
; RUN: grep "bl __misaligned_load" %t1.s | count 1
; RUN: grep ld16s %t1.s | count 2
; RUN: grep ldw %t1.s | count 2
; RUN: grep shl %t1.s | count 2
; RUN: grep shr %t1.s | count 1
; RUN: grep zext %t1.s | count 1
; RUN: grep "or " %t1.s | count 2

; Byte aligned load. Expands to call to __misaligned_load.
define i32 @align1(i32* %p) nounwind {
entry:
	%0 = load i32* %p, align 1		; <i32> [#uses=1]
	ret i32 %0
}

; Half word aligned load. Expands to two 16bit loads.
define i32 @align2(i32* %p) nounwind {
entry:
	%0 = load i32* %p, align 2		; <i32> [#uses=1]
	ret i32 %0
}

@a = global [5 x i8] zeroinitializer, align 4

; Constant offset from word aligned base. Expands to two 32bit loads.
define i32 @align3() nounwind {
entry:
	%0 = load i32* bitcast (i8* getelementptr ([5 x i8]* @a, i32 0, i32 1) to i32*), align 1
	ret i32 %0
}
