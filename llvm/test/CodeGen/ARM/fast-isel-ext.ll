; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=v7
; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=v7
; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=v7

; Can't test pre-ARMv6 Thumb because ARM FastISel currently only supports
; Thumb2. The ARMFastISel::ARMEmitIntExt code should work for Thumb by always
; using two shifts.

; Note that lsl, asr and lsr in Thumb are all encoded as 16-bit instructions
; and therefore must set flags. {{s?}} below denotes this, instead of
; duplicating tests.

; zext

define i8 @zext_1_8(i1 %a) nounwind ssp {
; v7-LABEL: zext_1_8:
; v7: and r0, r0, #1
  %r = zext i1 %a to i8
  ret i8 %r
}

define i16 @zext_1_16(i1 %a) nounwind ssp {
; v7-LABEL: zext_1_16:
; v7: and r0, r0, #1
  %r = zext i1 %a to i16
  ret i16 %r
}

define i32 @zext_1_32(i1 %a) nounwind ssp {
; v7-LABEL: zext_1_32:
; v7: and r0, r0, #1
  %r = zext i1 %a to i32
  ret i32 %r
}

define i16 @zext_8_16(i8 %a) nounwind ssp {
; v7-LABEL: zext_8_16:
; v7: and r0, r0, #255
  %r = zext i8 %a to i16
  ret i16 %r
}

define i32 @zext_8_32(i8 %a) nounwind ssp {
; v7-LABEL: zext_8_32:
; v7: and r0, r0, #255
  %r = zext i8 %a to i32
  ret i32 %r
}

define i32 @zext_16_32(i16 %a) nounwind ssp {
; v7-LABEL: zext_16_32:
; v7: uxth r0, r0
  %r = zext i16 %a to i32
  ret i32 %r
}

; sext

define i8 @sext_1_8(i1 %a) nounwind ssp {
; v7-LABEL: sext_1_8:
; v7: lsl{{s?}} r0, r0, #31
; v7: asr{{s?}} r0, r0, #31
  %r = sext i1 %a to i8
  ret i8 %r
}

define i16 @sext_1_16(i1 %a) nounwind ssp {
; v7-LABEL: sext_1_16:
; v7: lsl{{s?}} r0, r0, #31
; v7: asr{{s?}} r0, r0, #31
  %r = sext i1 %a to i16
  ret i16 %r
}

define i32 @sext_1_32(i1 %a) nounwind ssp {
; v7-LABEL: sext_1_32:
; v7: lsl{{s?}} r0, r0, #31
; v7: asr{{s?}} r0, r0, #31
  %r = sext i1 %a to i32
  ret i32 %r
}

define i16 @sext_8_16(i8 %a) nounwind ssp {
; v7-LABEL: sext_8_16:
; v7: sxtb r0, r0
  %r = sext i8 %a to i16
  ret i16 %r
}

define i32 @sext_8_32(i8 %a) nounwind ssp {
; v7-LABEL: sext_8_32:
; v7: sxtb r0, r0
  %r = sext i8 %a to i32
  ret i32 %r
}

define i32 @sext_16_32(i16 %a) nounwind ssp {
; v7-LABEL: sext_16_32:
; v7: sxth r0, r0
  %r = sext i16 %a to i32
  ret i32 %r
}
