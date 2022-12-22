; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that truncating stores are assembled properly.

; CHECK-LABEL: trunc_i8_i32:
; CHECK: i32.store8 0($0), $1{{$}}
define void @trunc_i8_i32(ptr %p, i32 %v) {
  %t = trunc i32 %v to i8
  store i8 %t, ptr %p
  ret void
}

; CHECK-LABEL: trunc_i16_i32:
; CHECK: i32.store16 0($0), $1{{$}}
define void @trunc_i16_i32(ptr %p, i32 %v) {
  %t = trunc i32 %v to i16
  store i16 %t, ptr %p
  ret void
}

; CHECK-LABEL: trunc_i8_i64:
; CHECK: i64.store8 0($0), $1{{$}}
define void @trunc_i8_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i8
  store i8 %t, ptr %p
  ret void
}

; CHECK-LABEL: trunc_i16_i64:
; CHECK: i64.store16 0($0), $1{{$}}
define void @trunc_i16_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i16
  store i16 %t, ptr %p
  ret void
}

; CHECK-LABEL: trunc_i32_i64:
; CHECK: i64.store32 0($0), $1{{$}}
define void @trunc_i32_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i32
  store i32 %t, ptr %p
  ret void
}
