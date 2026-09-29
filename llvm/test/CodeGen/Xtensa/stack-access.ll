; RUN: llc -mtriple=xtensa -O0 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=XTENSA

define i8 @loadi8(i8 %a) {
; XTENSA-LABEL: loadi8:
; XTENSA: s8i	a2, a1, 3
; XTENSA: l8ui	a2, a1, 3
; XTENSA: ret
  %b = alloca i8, align 1
  store i8 %a, ptr %b, align 1
  %1 = load i8, ptr %b, align 1
  ret i8 %1
}

define i16 @loadi16(i16 %a) {
; XTENSA-LABEL: loadi16:
; XTENSA: s16i	a2, a1, 2
; XTENSA: l16ui	a2, a1, 2
; XTENSA: ret
	%b = alloca i16, align 2
  store i16 %a, ptr %b, align 2
  %1 = load i16, ptr %b, align 2
	ret i16 %1
}

define i32 @loadi32(i32 %a) {
; XTENSA-LABEL: loadi32:
; XTENSA: s32i	a2, a1, 0
; XTENSA: l32i	a2, a1, 0
; XTENSA: ret
	%b = alloca i32, align 4
  store i32 %a, ptr %b, align 4
  %1 = load i32, ptr %b, align 4
	ret i32 %1
}
