; RUN: llc < %s -march=xcore | FileCheck %s

define i32 @load32(i32* %p, i32 %offset) nounwind {
entry:
; CHECK: load32:
; CHECK: ldw r0, r0[r1]
	%0 = getelementptr i32* %p, i32 %offset
	%1 = load i32* %0, align 4
	ret i32 %1
}

define i32 @load32_imm(i32* %p) nounwind {
entry:
; CHECK: load32_imm:
; CHECK: ldw r0, r0[11]
	%0 = getelementptr i32* %p, i32 11
	%1 = load i32* %0, align 4
	ret i32 %1
}

define i32 @load16(i16* %p, i32 %offset) nounwind {
entry:
; CHECK: load16:
; CHECK: ld16s r0, r0[r1]
; CHECK-NOT: sext
	%0 = getelementptr i16* %p, i32 %offset
	%1 = load i16* %0, align 2
	%2 = sext i16 %1 to i32
	ret i32 %2
}

define i32 @load8(i8* %p, i32 %offset) nounwind {
entry:
; CHECK: load8:
; CHECK: ld8u r0, r0[r1]
; CHECK-NOT: zext
	%0 = getelementptr i8* %p, i32 %offset
	%1 = load i8* %0, align 1
	%2 = zext i8 %1 to i32
	ret i32 %2
}
