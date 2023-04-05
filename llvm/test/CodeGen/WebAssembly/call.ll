; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -mattr=+sign-ext,+simd128 | FileCheck --check-prefixes=CHECK,NO-TAIL %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 -mattr=+sign-ext,+simd128 | FileCheck --check-prefixes=CHECK,NO-TAIL %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -mattr=+sign-ext,+simd128,+tail-call | FileCheck --check-prefixes=CHECK,SLOW-TAIL %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 -mattr=+sign-ext,+simd128,+tail-call | FileCheck --check-prefixes=CHECK,FAST-TAIL %s

; Test that basic call operations assemble as expected.

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
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, i32_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_i32_nullary() {
  %r = call i32 @i32_nullary()
  ret i32 %r
}

; CHECK-LABEL: call_i64_nullary:
; CHECK-NEXT: .functype call_i64_nullary () -> (i64){{$}}
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, i64_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @call_i64_nullary() {
  %r = call i64 @i64_nullary()
  ret i64 %r
}

; CHECK-LABEL: call_float_nullary:
; CHECK-NEXT: .functype call_float_nullary () -> (f32){{$}}
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, float_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @call_float_nullary() {
  %r = call float @float_nullary()
  ret float %r
}

; CHECK-LABEL: call_double_nullary:
; CHECK-NEXT: .functype call_double_nullary () -> (f64){{$}}
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, double_nullary{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @call_double_nullary() {
  %r = call double @double_nullary()
  ret double %r
}

; CHECK-LABEL: call_v128_nullary:
; CHECK-NEXT: .functype call_v128_nullary () -> (v128){{$}}
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, v128_nullary{{$}}
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
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, i32_unary, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_i32_unary(i32 %a) {
  %r = call i32 @i32_unary(i32 %a)
  ret i32 %r
}

; CHECK-LABEL: call_i32_binary:
; CHECK-NEXT: .functype call_i32_binary (i32, i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: {{^}} call $push[[NUM:[0-9]+]]=, i32_binary, $pop[[L0]], $pop[[L1]]{{$}}
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
define void @call_indirect_void(ptr %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: call_indirect_i32:
; CHECK-NEXT: .functype call_indirect_i32 (i32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_indirect_i32(ptr %callee) {
  %t = call i32 %callee()
  ret i32 %t
}

; CHECK-LABEL: call_indirect_i64:
; CHECK-NEXT: .functype call_indirect_i64 (i32) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @call_indirect_i64(ptr %callee) {
  %t = call i64 %callee()
  ret i64 %t
}

; CHECK-LABEL: call_indirect_float:
; CHECK-NEXT: .functype call_indirect_float (i32) -> (f32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @call_indirect_float(ptr %callee) {
  %t = call float %callee()
  ret float %t
}

; CHECK-LABEL: call_indirect_double:
; CHECK-NEXT: .functype call_indirect_double (i32) -> (f64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @call_indirect_double(ptr %callee) {
  %t = call double %callee()
  ret double %t
}

; CHECK-LABEL: call_indirect_v128:
; CHECK-NEXT: .functype call_indirect_v128 (i32) -> (v128){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define <16 x i8> @call_indirect_v128(ptr %callee) {
  %t = call <16 x i8> %callee()
  ret <16 x i8> %t
}

; CHECK-LABEL: call_indirect_arg:
; CHECK-NEXT: .functype call_indirect_arg (i32, i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_arg(ptr %callee, i32 %arg) {
  call void %callee(i32 %arg)
  ret void
}

; CHECK-LABEL: call_indirect_arg_2:
; CHECK-NEXT: .functype call_indirect_arg_2 (i32, i32, i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: drop $pop[[NUM]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_arg_2(ptr %callee, i32 %arg, i32 %arg2) {
  call i32 %callee(i32 %arg, i32 %arg2)
  ret void
}

; CHECK-LABEL: tail_call_void_nullary:
; CHECK-NEXT: .functype tail_call_void_nullary () -> (){{$}}
; NO-TAIL-NEXT: {{^}} call void_nullary{{$}}
; NO-TAIL-NEXT: return{{$}}
; SLOW-TAIL-NEXT: {{^}} return_call void_nullary{{$}}
; FAST-TAIL-NEXT: {{^}} call void_nullary{{$}}
; FAST-TAIL-NEXT: return{{$}}
define void @tail_call_void_nullary() {
  tail call void @void_nullary()
  ret void
}

; CHECK-LABEL: fastcc_tail_call_void_nullary:
; CHECK-NEXT: .functype fastcc_tail_call_void_nullary () -> (){{$}}
; NO-TAIL-NEXT: {{^}} call void_nullary{{$}}
; NO-TAIL-NEXT: return{{$}}
; SLOW-TAIL-NEXT: {{^}} return_call void_nullary{{$}}
; FAST-TAIL-NEXT: {{^}} call void_nullary{{$}}
; FAST-TAIL-NEXT: return{{$}}
define void @fastcc_tail_call_void_nullary() {
  tail call fastcc void @void_nullary()
  ret void
}

; CHECK-LABEL: coldcc_tail_call_void_nullary:
; CHECK-NEXT: .functype coldcc_tail_call_void_nullary () -> (){{$}}
; NO-TAIL-NEXT: {{^}} call void_nullary{{$}}
; NO-TAIL-NEXT: return{{$}}
; SLOW-TAIL-NEXT: {{^}} return_call void_nullary{{$}}
; FAST-TAIL-NEXT: {{^}} call void_nullary{{$}}
; FAST-TAIL-NEXT: return{{$}}
define void @coldcc_tail_call_void_nullary() {
  tail call coldcc void @void_nullary()
  ret void
}

; CHECK-LABEL: call_constexpr:
; CHECK-NEXT: .functype call_constexpr () -> (){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: call .Lvararg_func_bitcast, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const	$push[[L3:[0-9]+]]=, void_nullary{{$}}
; CHECK-NEXT: i32.const	$push[[L2:[0-9]+]]=, other_void_nullary{{$}}
; CHECK-NEXT: i32.add 	$push[[L4:[0-9]+]]=, $pop[[L3]], $pop[[L2]]{{$}}
; CHECK-NEXT: call_indirect	$pop[[L4]]{{$}}
; CHECK-NEXT: call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
declare void @vararg_func(...)
declare void @other_void_nullary()
define void @call_constexpr() {
bb0:
  call void @vararg_func(i32 2, i32 3)
  br label %bb1
bb1:
  call void getelementptr (i8, ptr @void_nullary, i32 ptrtoint (ptr @other_void_nullary to i32))()
  br label %bb2
bb2:
  call void inttoptr (i32 ptrtoint (ptr @void_nullary to i32) to ptr)()
  ret void
}

; Allocas should be lowered to call_indirects.
; CHECK-LABEL: call_indirect_alloca:
; CHECK:      local.tee  $push{{.*}}=, [[L0:[0-9]+]]
; CHECK-NEXT: global.set  __stack_pointer
; CHECK-NEXT: local.get  $push{{.*}}=, [[L0]]
; CHECK-NEXT: i32.const  $push{{.*}}=, 12
; CHECK-NEXT: i32.add
; CHECK-NEXT: call_indirect  $pop{{.*}}
define void @call_indirect_alloca() {
entry:
  %ptr = alloca i32, align 4
  call void %ptr()
  ret void
}

; TODO: test the following:
;  - More argument combinations.
;  - Tail call.
;  - Interesting returns (struct, multiple).
;  - Vararg.
