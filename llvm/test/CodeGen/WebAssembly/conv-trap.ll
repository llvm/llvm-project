; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=-nontrapping-fptoint | FileCheck %s

; Test that basic conversion operations assemble as expected using
; the trapping opcodes and explicit code to suppress the trapping.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: i32_trunc_s_f32:
; CHECK-NEXT: .functype i32_trunc_s_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f32.abs $push[[ABS:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: f32.const $push[[LIMIT:[0-9]+]]=, 0x1p31{{$}}
; CHECK-NEXT: f32.lt $push[[LT:[0-9]+]]=, $pop[[ABS]], $pop[[LIMIT]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[LT]]{{$}}
; CHECK-NEXT: i32.const $push[[ALT:[0-9]+]]=, -2147483648{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i32.trunc_s/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_s_f32(float %x) {
  %a = fptosi float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f32:
; CHECK-NEXT: .functype i32_trunc_u_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f32.const $push[[LIMIT:[0-9]+]]=, 0x1p32{{$}}
; CHECK-NEXT: f32.lt $push[[LT:[0-9]+]]=, $0, $pop[[LIMIT]]{{$}}
; CHECK-NEXT: f32.const $push[[ZERO:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: f32.ge $push[[GE:[0-9]+]]=, $0, $pop[[ZERO]]{{$}}
; CHECK-NEXT: i32.and $push[[AND:[0-9]+]]=, $pop[[LT]], $pop[[GE]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[AND]]{{$}}
; CHECK-NEXT: i32.const $push[[ALT:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i32.trunc_u/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_u_f32(float %x) {
  %a = fptoui float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_s_f64:
; CHECK-NEXT: .functype i32_trunc_s_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f64.abs $push[[ABS:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: f64.const $push[[LIMIT:[0-9]+]]=, 0x1p31{{$}}
; CHECK-NEXT: f64.lt $push[[LT:[0-9]+]]=, $pop[[ABS]], $pop[[LIMIT]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[LT]]{{$}}
; CHECK-NEXT: i32.const $push[[ALT:[0-9]+]]=, -2147483648{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i32.trunc_s/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_s_f64(double %x) {
  %a = fptosi double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f64:
; CHECK-NEXT: .functype i32_trunc_u_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f64.const $push[[LIMIT:[0-9]+]]=, 0x1p32{{$}}
; CHECK-NEXT: f64.lt $push[[LT:[0-9]+]]=, $0, $pop[[LIMIT]]{{$}}
; CHECK-NEXT: f64.const $push[[ZERO:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: f64.ge $push[[GE:[0-9]+]]=, $0, $pop[[ZERO]]{{$}}
; CHECK-NEXT: i32.and $push[[AND:[0-9]+]]=, $pop[[LT]], $pop[[GE]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[AND]]{{$}}
; CHECK-NEXT: i32.const $push[[ALT:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i32.trunc_u/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_u_f64(double %x) {
  %a = fptoui double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_trunc_s_f32:
; CHECK-NEXT: .functype i64_trunc_s_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f32.abs $push[[ABS:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: f32.const $push[[LIMIT:[0-9]+]]=, 0x1p63{{$}}
; CHECK-NEXT: f32.lt $push[[LT:[0-9]+]]=, $pop[[ABS]], $pop[[LIMIT]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[LT]]{{$}}
; CHECK-NEXT: i64.const $push[[ALT:[0-9]+]]=, -9223372036854775808{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i64.trunc_s/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_s_f32(float %x) {
  %a = fptosi float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f32:
; CHECK-NEXT: .functype i64_trunc_u_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f32.const $push[[LIMIT:[0-9]+]]=, 0x1p64{{$}}
; CHECK-NEXT: f32.lt $push[[LT:[0-9]+]]=, $0, $pop[[LIMIT]]{{$}}
; CHECK-NEXT: f32.const $push[[ZERO:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: f32.ge $push[[GE:[0-9]+]]=, $0, $pop[[ZERO]]{{$}}
; CHECK-NEXT: i32.and $push[[AND:[0-9]+]]=, $pop[[LT]], $pop[[GE]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[AND]]{{$}}
; CHECK-NEXT: i64.const $push[[ALT:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i64.trunc_u/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_u_f32(float %x) {
  %a = fptoui float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_s_f64:
; CHECK-NEXT: .functype i64_trunc_s_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f64.abs $push[[ABS:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: f64.const $push[[LIMIT:[0-9]+]]=, 0x1p63{{$}}
; CHECK-NEXT: f64.lt $push[[LT:[0-9]+]]=, $pop[[ABS]], $pop[[LIMIT]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[LT]]{{$}}
; CHECK-NEXT: i64.const $push[[ALT:[0-9]+]]=, -9223372036854775808{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i64.trunc_s/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_s_f64(double %x) {
  %a = fptosi double %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f64:
; CHECK-NEXT: .functype i64_trunc_u_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: block
; CHECK-NEXT: f64.const $push[[LIMIT:[0-9]+]]=, 0x1p64{{$}}
; CHECK-NEXT: f64.lt $push[[LT:[0-9]+]]=, $0, $pop[[LIMIT]]{{$}}
; CHECK-NEXT: f64.const $push[[ZERO:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: f64.ge $push[[GE:[0-9]+]]=, $0, $pop[[ZERO]]{{$}}
; CHECK-NEXT: i32.and $push[[AND:[0-9]+]]=, $pop[[LT]], $pop[[GE]]{{$}}
; CHECK-NEXT: br_if 0, $pop[[AND]]{{$}}
; CHECK-NEXT: i64.const $push[[ALT:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[ALT]]{{$}}
; CHECK-NEXT: BB
; CHECK-NEXT: end_block
; CHECK-NEXT: i64.trunc_u/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_u_f64(double %x) {
  %a = fptoui double %x to i64
  ret i64 %a
}
