; RUN: llc < %s -march=xcore | FileCheck %s
%0 = type { i32, i32 }

declare i32 @llvm.xcore.bitrev(i32)
declare i32 @llvm.xcore.crc32(i32, i32, i32)
declare %0 @llvm.xcore.crc8(i32, i32, i32)
declare i32 @llvm.xcore.zext(i32, i32)
declare i32 @llvm.xcore.sext(i32, i32)
declare i32 @llvm.xcore.geted()
declare i32 @llvm.xcore.getet()

define i32 @bitrev(i32 %val) {
; CHECK: bitrev:
; CHECK: bitrev r0, r0
	%result = call i32 @llvm.xcore.bitrev(i32 %val)
	ret i32 %result
}

define i32 @crc32(i32 %crc, i32 %data, i32 %poly) {
; CHECK: crc32:
; CHECK: crc32 r0, r1, r2
	%result = call i32 @llvm.xcore.crc32(i32 %crc, i32 %data, i32 %poly)
	ret i32 %result
}

define %0 @crc8(i32 %crc, i32 %data, i32 %poly) {
; CHECK: crc8:
; CHECK: crc8 r0, r1, r1, r2
	%result = call %0 @llvm.xcore.crc8(i32 %crc, i32 %data, i32 %poly)
	ret %0 %result
}

define i32 @zext(i32 %a, i32 %b) {
; CHECK: zext:
; CHECK: zext r0, r1
	%result = call i32 @llvm.xcore.zext(i32 %a, i32 %b)
	ret i32 %result
}

define i32 @zexti(i32 %a) {
; CHECK: zexti:
; CHECK: zext r0, 4
	%result = call i32 @llvm.xcore.zext(i32 %a, i32 4)
	ret i32 %result
}

define i32 @sext(i32 %a, i32 %b) {
; CHECK: sext:
; CHECK: sext r0, r1
	%result = call i32 @llvm.xcore.sext(i32 %a, i32 %b)
	ret i32 %result
}

define i32 @sexti(i32 %a) {
; CHECK: sexti:
; CHECK: sext r0, 4
	%result = call i32 @llvm.xcore.sext(i32 %a, i32 4)
	ret i32 %result
}

define i32 @geted() {
; CHECK: geted:
; CHECK: get r11, ed
; CHECK-NEXT: mov r0, r11
	%result = call i32 @llvm.xcore.geted()
	ret i32 %result
}

define i32 @getet() {
; CHECK: getet:
; CHECK: get r11, et
; CHECK-NEXT: mov r0, r11
	%result = call i32 @llvm.xcore.getet()
	ret i32 %result
}
