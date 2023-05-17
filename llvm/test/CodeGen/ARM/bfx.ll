; RUN: llc -mtriple=arm-eabi -mattr=+v7 %s -o - | FileCheck %s

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

; rdar://12870177
define i32 @ubfx_opt(ptr nocapture %ctx, i32 %x) nounwind readonly ssp {
entry:
; CHECK: ubfx_opt
; CHECK: lsr [[REG1:(lr|r[0-9]+)]], r1, #24
; CHECK: ldr {{lr|r[0-9]+}}, [r0, [[REG1]], lsl #2]
; CHECK: ubfx [[REG2:(lr|r[0-9]+)]], r1, #16, #8
; CHECK: ldr {{lr|r[0-9]+}}, [r0, [[REG2]], lsl #2]
; CHECK: ubfx [[REG3:(lr|r[0-9]+)]], r1, #8, #8
; CHECK: ldr {{lr|r[0-9]+}}, [r0, [[REG3]], lsl #2]
  %and = lshr i32 %x, 8
  %shr = and i32 %and, 255
  %and1 = lshr i32 %x, 16
  %shr2 = and i32 %and1, 255
  %shr4 = lshr i32 %x, 24
  %arrayidx = getelementptr inbounds i32, ptr %ctx, i32 %shr4
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds i32, ptr %ctx, i32 %shr2
  %1 = load i32, ptr %arrayidx5, align 4
  %add = add i32 %1, %0
  %arrayidx6 = getelementptr inbounds i32, ptr %ctx, i32 %shr
  %2 = load i32, ptr %arrayidx6, align 4
  %add7 = add i32 %add, %2
  ret i32 %add7
}

define i32 @ubfx3(i32 %a) {
; CHECK: ubfx3
; CHECK: ubfx r0, r0, #11, #1
	%t1 = and i32 %a, 2048
	%t2 = lshr i32 %t1, 11
	ret i32 %t2
}

define i32 @ubfx4(i32 %a) {
; CHECK: ubfx4
; CHECK: ubfx r0, r0, #7, #3
	%t1 = and i32 %a, 896
	%t2 = lshr i32 %t1, 7
	ret i32 %t2
}
