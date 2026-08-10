; RUN: llc < %s -mtriple=xcore | FileCheck %s

define i32 @load32(ptr %p, i32 %offset) nounwind {
entry:
; CHECK-LABEL: load32:
; CHECK: ldw r0, r0[r1]
	%0 = getelementptr i32, ptr %p, i32 %offset
	%1 = load i32, ptr %0, align 4
	ret i32 %1
}

define i32 @load32_imm(ptr %p) nounwind {
entry:
; CHECK-LABEL: load32_imm:
; CHECK: ldw r0, r0[11]
	%0 = getelementptr i32, ptr %p, i32 11
	%1 = load i32, ptr %0, align 4
	ret i32 %1
}

define i32 @load16(ptr %p, i32 %offset) nounwind {
entry:
; CHECK-LABEL: load16:
; CHECK: ld16s r0, r0[r1]
; CHECK-NOT: sext
	%0 = getelementptr i16, ptr %p, i32 %offset
	%1 = load i16, ptr %0, align 2
	%2 = sext i16 %1 to i32
	ret i32 %2
}

define i32 @load8(ptr %p, i32 %offset) nounwind {
entry:
; CHECK-LABEL: load8:
; CHECK: ld8u r0, r0[r1]
; CHECK-NOT: zext
	%0 = getelementptr i8, ptr %p, i32 %offset
	%1 = load i8, ptr %0, align 1
	%2 = zext i8 %1 to i32
	ret i32 %2
}

@GConst = internal constant i32 42
define i32 @load_cp() nounwind {
entry:
; CHECK-LABEL: load_cp:
; CHECK: ldw r0, cp[GConst]
  %0 = load i32, ptr @GConst
  ret i32 %0
}
