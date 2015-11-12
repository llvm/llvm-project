; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that extending loads are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: sext_i8_i32:
; CHECK: i32.load8_s $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i32 @sext_i8_i32(i8 *%p) {
  %v = load i8, i8* %p
  %e = sext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i8_i32:
; CHECK: i32.load8_u $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i32 @zext_i8_i32(i8 *%p) {
  %v = load i8, i8* %p
  %e = zext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i16_i32:
; CHECK: i32.load16_s $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i32 @sext_i16_i32(i16 *%p) {
  %v = load i16, i16* %p
  %e = sext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i16_i32:
; CHECK: i32.load16_u $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i32 @zext_i16_i32(i16 *%p) {
  %v = load i16, i16* %p
  %e = zext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i8_i64:
; CHECK: i64.load8_s $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i64 @sext_i8_i64(i8 *%p) {
  %v = load i8, i8* %p
  %e = sext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i8_i64:
; CHECK: i64.load8_u $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i64 @zext_i8_i64(i8 *%p) {
  %v = load i8, i8* %p
  %e = zext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i16_i64:
; CHECK: i64.load16_s $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i64 @sext_i16_i64(i16 *%p) {
  %v = load i16, i16* %p
  %e = sext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i16_i64:
; CHECK: i64.load16_u $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i64 @zext_i16_i64(i16 *%p) {
  %v = load i16, i16* %p
  %e = zext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i32_i64:
; CHECK: i64.load32_s $push, (get_local 0){{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
define i64 @sext_i32_i64(i32 *%p) {
  %v = load i32, i32* %p
  %e = sext i32 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i32_i64:
; CHECK: i64.load32_u $push, (get_local 0){{$}}
; CHECK: set_local 1, $pop{{$}}
define i64 @zext_i32_i64(i32 *%p) {
  %v = load i32, i32* %p
  %e = zext i32 %v to i64
  ret i64 %e
}
