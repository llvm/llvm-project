; RUN: llc < %s -march=xcore | FileCheck %s

define void @store32(ptr %p, i32 %offset, i32 %val) nounwind {
entry:
; CHECK-LABEL: store32:
; CHECK: stw r2, r0[r1]
	%0 = getelementptr i32, ptr %p, i32 %offset
	store i32 %val, ptr %0, align 4
	ret void
}

define void @store32_imm(ptr %p, i32 %val) nounwind {
entry:
; CHECK-LABEL: store32_imm:
; CHECK: stw r1, r0[11]
	%0 = getelementptr i32, ptr %p, i32 11
	store i32 %val, ptr %0, align 4
	ret void
}

define void @store16(ptr %p, i32 %offset, i16 %val) nounwind {
entry:
; CHECK-LABEL: store16:
; CHECK: st16 r2, r0[r1]
	%0 = getelementptr i16, ptr %p, i32 %offset
	store i16 %val, ptr %0, align 2
	ret void
}

define void @store8(ptr %p, i32 %offset, i8 %val) nounwind {
entry:
; CHECK-LABEL: store8:
; CHECK: st8 r2, r0[r1]
	%0 = getelementptr i8, ptr %p, i32 %offset
	store i8 %val, ptr %0, align 1
	ret void
}
