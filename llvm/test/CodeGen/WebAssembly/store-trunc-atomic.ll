; RUN: llc < %s --mtriple=wasm32-unknown-unknown -mattr=+atomics,+sign-ext -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -mattr=+atomics,+sign-ext -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that truncating stores are assembled properly.

; CHECK-LABEL: trunc_i8_i32:
; CHECK: i32.atomic.store8 0($0), $1{{$}}
define void @trunc_i8_i32(ptr %p, i32 %v) {
  %t = trunc i32 %v to i8
  store atomic i8 %t, ptr %p seq_cst, align 1
  ret void
}

; CHECK-LABEL: trunc_i16_i32:
; CHECK: i32.atomic.store16 0($0), $1{{$}}
define void @trunc_i16_i32(ptr %p, i32 %v) {
  %t = trunc i32 %v to i16
  store atomic i16 %t, ptr %p seq_cst, align 2
  ret void
}

; CHECK-LABEL: trunc_i8_i64:
; CHECK: i64.atomic.store8 0($0), $1{{$}}
define void @trunc_i8_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i8
  store atomic i8 %t, ptr %p seq_cst, align 1
  ret void
}

; CHECK-LABEL: trunc_i16_i64:
; CHECK: i64.atomic.store16 0($0), $1{{$}}
define void @trunc_i16_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i16
  store atomic i16 %t, ptr %p seq_cst, align 2
  ret void
}

; CHECK-LABEL: trunc_i32_i64:
; CHECK: i64.atomic.store32 0($0), $1{{$}}
define void @trunc_i32_i64(ptr %p, i64 %v) {
  %t = trunc i64 %v to i32
  store atomic i32 %t, ptr %p seq_cst, align 4
  ret void
}
