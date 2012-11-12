; RUN: llc < %s -march=xcore | FileCheck %s

define void @store32(i32* %p, i32 %offset, i32 %val) nounwind {
entry:
; CHECK: store32:
; CHECK: stw r2, r0[r1]
	%0 = getelementptr i32* %p, i32 %offset
	store i32 %val, i32* %0, align 4
	ret void
}

define void @store32_imm(i32* %p, i32 %val) nounwind {
entry:
; CHECK: store32_imm:
; CHECK: stw r1, r0[11]
	%0 = getelementptr i32* %p, i32 11
	store i32 %val, i32* %0, align 4
	ret void
}

define void @store16(i16* %p, i32 %offset, i16 %val) nounwind {
entry:
; CHECK: store16:
; CHECK: st16 r2, r0[r1]
	%0 = getelementptr i16* %p, i32 %offset
	store i16 %val, i16* %0, align 2
	ret void
}

define void @store8(i8* %p, i32 %offset, i8 %val) nounwind {
entry:
; CHECK: store8:
; CHECK: st8 r2, r0[r1]
	%0 = getelementptr i8* %p, i32 %offset
	store i8 %val, i8* %0, align 1
	ret void
}
