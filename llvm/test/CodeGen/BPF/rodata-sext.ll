; RUN: llc -mtriple=bpfel -mcpu=v3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-V3
; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-V4

; Constant SEXTLOAD folding must preserve sign extension.
; v3 emits the sign extension separately, while v4 folds the SEXTLOAD itself.

@byte = constant { i8 } { i8 -1 }
@half = constant { i16 } { i16 -2 }
@word = constant { i32 } { i32 -3 }

define i32 @sext_i8_i32() {
; CHECK-V3-LABEL: sext_i8_i32:
; CHECK-V3:       w0 = 255
; CHECK-V3-NEXT:  w0 <<= 24
; CHECK-V3-NEXT:  w0 s>>= 24
; CHECK-V3-NEXT:  exit
; CHECK-V4-LABEL: sext_i8_i32:
; CHECK-V4:       w0 = -1
; CHECK-V4-NEXT:  exit
  %v = load i8, ptr @byte
  %ext = sext i8 %v to i32
  ret i32 %ext
}

define i64 @sext_i8_i64() {
; CHECK-V4-LABEL: sext_i8_i64:
; CHECK-V4:       r0 = -1
; CHECK-V4-NEXT:  exit
  %v = load i8, ptr @byte
  %ext = sext i8 %v to i64
  ret i64 %ext
}

define i64 @sext_i16_i64() {
; CHECK-V4-LABEL: sext_i16_i64:
; CHECK-V4:       r0 = -2
; CHECK-V4-NEXT:  exit
  %v = load i16, ptr @half
  %ext = sext i16 %v to i64
  ret i64 %ext
}

define i64 @sext_i32_i64() {
; CHECK-V4-LABEL: sext_i32_i64:
; CHECK-V4:       r0 = -3
; CHECK-V4-NEXT:  exit
  %v = load i32, ptr @word
  %ext = sext i32 %v to i64
  ret i64 %ext
}
