; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -mattr=+sign-ext,+simd128 | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 -mattr=+sign-ext,+simd128 | FileCheck %s

; Test that basic call operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @i32_nullary()
declare i32 @i32_unary(i32)
declare i32 @i32_binary(i32, i32)
declare i64 @i64_nullary()
declare float @float_nullary()
declare double @double_nullary()
declare <16 x i8> @v128_nullary()
declare void @void_nullary()

; CHECK-LABEL: call_i32_nullary:
; CHECK-NEXT: .functype call_i32_nullary () -> (i32){{$}}
; CHECK-NEXT: {{^}} i32.call $push[[NUM:[0-9]+]]=, i32_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_i32_nullary() {
  %r = call i32 @i32_nullary()
  ret i32 %r
}

; CHECK-LABEL: call_i64_nullary:
; CHECK-NEXT: .functype call_i64_nullary () -> (i64){{$}}
; CHECK-NEXT: {{^}} i64.call $push[[NUM:[0-9]+]]=, i64_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @call_i64_nullary() {
  %r = call i64 @i64_nullary()
  ret i64 %r
}

; CHECK-LABEL: call_float_nullary:
; CHECK-NEXT: .functype call_float_nullary () -> (f32){{$}}
; CHECK-NEXT: {{^}} f32.call $push[[NUM:[0-9]+]]=, float_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @call_float_nullary() {
  %r = call float @float_nullary()
  ret float %r
}

; CHECK-LABEL: call_double_nullary:
; CHECK-NEXT: .functype call_double_nullary () -> (f64){{$}}
; CHECK-NEXT: {{^}} f64.call $push[[NUM:[0-9]+]]=, double_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @call_double_nullary() {
  %r = call double @double_nullary()
  ret double %r
}

; CHECK-LABEL: call_v128_nullary:
; CHECK-NEXT: .functype call_v128_nullary () -> (v128){{$}}
; CHECK-NEXT: {{^}} v128.call $push[[NUM:[0-9]+]]=, v128_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define <16 x i8> @call_v128_nullary() {
  %r = call <16 x i8> @v128_nullary()
  ret <16 x i8> %r
}

; CHECK-LABEL: call_void_nullary:
; CHECK-NEXT: .functype call_void_nullary () -> (){{$}}
; CHECK-NEXT: {{^}} call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @call_void_nullary() {
  call void @void_nullary()
  ret void
}

; CHECK-LABEL: call_i32_unary:
; CHECK-NEXT: .functype call_i32_unary (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} i32.call $push[[NUM:[0-9]+]]=, i32_unary, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_i32_unary(i32 %a) {
  %r = call i32 @i32_unary(i32 %a)
  ret i32 %r
}

; CHECK-LABEL: call_i32_binary:
; CHECK-NEXT: .functype call_i32_binary (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: {{^}} i32.call $push[[NUM:[0-9]+]]=, i32_binary, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_i32_binary(i32 %a, i32 %b) {
  %r = call i32 @i32_binary(i32 %a, i32 %b)
  ret i32 %r
}

; CHECK-LABEL: call_indirect_void:
; CHECK-NEXT: .functype call_indirect_void (i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $pop[[L0]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_void(void ()* %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: call_indirect_i32:
; CHECK-NEXT: .functype call_indirect_i32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} i32.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_indirect_i32(i32 ()* %callee) {
  %t = call i32 %callee()
  ret i32 %t
}

; CHECK-LABEL: call_indirect_i64:
; CHECK-NEXT: .functype call_indirect_i64 (i32) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} i64.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @call_indirect_i64(i64 ()* %callee) {
  %t = call i64 %callee()
  ret i64 %t
}

; CHECK-LABEL: call_indirect_float:
; CHECK-NEXT: .functype call_indirect_float (i32) -> (f32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} f32.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @call_indirect_float(float ()* %callee) {
  %t = call float %callee()
  ret float %t
}

; CHECK-LABEL: call_indirect_double:
; CHECK-NEXT: .functype call_indirect_double (i32) -> (f64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} f64.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @call_indirect_double(double ()* %callee) {
  %t = call double %callee()
  ret double %t
}

; CHECK-LABEL: call_indirect_v128:
; CHECK-NEXT: .functype call_indirect_v128 (i32) -> (v128){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} v128.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define <16 x i8> @call_indirect_v128(<16 x i8> ()* %callee) {
  %t = call <16 x i8> %callee()
  ret <16 x i8> %t
}

; CHECK-LABEL: call_indirect_arg:
; CHECK-NEXT: .functype call_indirect_arg (i32, i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_arg(void (i32)* %callee, i32 %arg) {
  call void %callee(i32 %arg)
  ret void
}

; CHECK-LABEL: call_indirect_arg_2:
; CHECK-NEXT: .functype call_indirect_arg_2 (i32, i32, i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} i32.call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: drop $pop[[NUM]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_arg_2(i32 (i32, i32)* %callee, i32 %arg, i32 %arg2) {
  call i32 %callee(i32 %arg, i32 %arg2)
  ret void
}

; CHECK-LABEL: tail_call_void_nullary:
; CHECK-NEXT: .functype tail_call_void_nullary () -> (){{$}}
; CHECK-NEXT: {{^}} call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @tail_call_void_nullary() {
  tail call void @void_nullary()
  ret void
}

; CHECK-LABEL: fastcc_tail_call_void_nullary:
; CHECK-NEXT: .functype fastcc_tail_call_void_nullary () -> (){{$}}
; CHECK-NEXT: {{^}} call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @fastcc_tail_call_void_nullary() {
  tail call fastcc void @void_nullary()
  ret void
}

; CHECK-LABEL: coldcc_tail_call_void_nullary:
; CHECK-NEXT: .functype coldcc_tail_call_void_nullary () -> (){{$}}
; CHECK-NEXT: {{^}} call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @coldcc_tail_call_void_nullary() {
  tail call coldcc void @void_nullary()
  ret void
}

; CHECK-LABEL: call_constexpr:
; CHECK-NEXT: .functype call_constexpr () -> (){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: call .Lvararg_func_bitcast, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: call other_void_nullary{{$}}
; CHECK-NEXT: call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
declare void @vararg_func(...)
declare void @other_void_nullary()
define void @call_constexpr() {
bb0:
  call void bitcast (void (...)* @vararg_func to void (i32, i32)*)(i32 2, i32 3)
  br label %bb1
bb1:
  call void select (i1 0, void ()* @void_nullary, void ()* @other_void_nullary)()
  br label %bb2
bb2:
  call void inttoptr (i32 ptrtoint (void ()* @void_nullary to i32) to void ()*)()
  ret void
}

; TODO: test the following:
;  - More argument combinations.
;  - Tail call.
;  - Interesting returns (struct, multiple).
;  - Vararg.
