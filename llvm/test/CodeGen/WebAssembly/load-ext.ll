; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that extending loads are assembled properly.

; CHECK-LABEL: sext_i8_i32:
; CHECK: i32.load8_s $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sext_i8_i32(ptr %p) {
  %v = load i8, ptr %p
  %e = sext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i8_i32:
; CHECK: i32.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @zext_i8_i32(ptr %p) {
  %v = load i8, ptr %p
  %e = zext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i16_i32:
; CHECK: i32.load16_s $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sext_i16_i32(ptr %p) {
  %v = load i16, ptr %p
  %e = sext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i16_i32:
; CHECK: i32.load16_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @zext_i16_i32(ptr %p) {
  %v = load i16, ptr %p
  %e = zext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i8_i64:
; CHECK: i64.load8_s $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sext_i8_i64(ptr %p) {
  %v = load i8, ptr %p
  %e = sext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i8_i64:
; CHECK: i64.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @zext_i8_i64(ptr %p) {
  %v = load i8, ptr %p
  %e = zext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i16_i64:
; CHECK: i64.load16_s $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sext_i16_i64(ptr %p) {
  %v = load i16, ptr %p
  %e = sext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i16_i64:
; CHECK: i64.load16_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @zext_i16_i64(ptr %p) {
  %v = load i16, ptr %p
  %e = zext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i32_i64:
; CHECK: i64.load32_s $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sext_i32_i64(ptr %p) {
  %v = load i32, ptr %p
  %e = sext i32 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i32_i64:
; CHECK: i64.load32_u $push0=, 0($0){{$}}
; CHECK: return $pop0{{$}}
define i64 @zext_i32_i64(ptr %p) {
  %v = load i32, ptr %p
  %e = zext i32 %v to i64
  ret i64 %e
}
