; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @sbfx1(i32 %a) {
; CHECK: sbfx1
; CHECK: sbfx r0, r0, #7, #11
	%t1 = lshr i32 %a, 7
	%t2 = trunc i32 %t1 to i11
	%t3 = sext i11 %t2 to i32
	ret i32 %t3
}

define i32 @ubfx1(i32 %a) {
; CHECK: ubfx1
; CHECK: ubfx r0, r0, #7, #11
	%t1 = lshr i32 %a, 7
	%t2 = trunc i32 %t1 to i11
	%t3 = zext i11 %t2 to i32
	ret i32 %t3
}

define i32 @ubfx2(i32 %a) {
; CHECK: ubfx2
; CHECK: ubfx r0, r0, #7, #11
	%t1 = lshr i32 %a, 7
	%t2 = and i32 %t1, 2047
	ret i32 %t2
}

