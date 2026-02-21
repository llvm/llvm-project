; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -mattr=+sign-ext,+simd128 | FileCheck --check-prefixes=CHECK,SLOW %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 -mattr=+sign-ext,+simd128 | FileCheck --check-prefixes=CHECK,FAST %s
;
; Test that the relavent (differing) subsets of `call.ll` function correctly under Wasm64

target triple = "wasm64-unknown-unknown"

declare void @void_nullary()

; CHECK-LABEL: call_void_nullary:
; CHECK-NEXT: .functype call_void_nullary () -> (){{$}}
; CHECK-NEXT: {{^}} call void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @call_void_nullary() {
  call void @void_nullary()
  ret void
}

; CHECK-LABEL: call_indirect_void:
; CHECK-NEXT: .functype call_indirect_void (i64) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $pop[[L0]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_void(ptr %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: call_indirect_i32:
; CHECK-NEXT: .functype call_indirect_i32 (i64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @call_indirect_i32(ptr %callee) {
  %t = call i32 %callee()
  ret i32 %t
}

; CHECK-LABEL: call_indirect_i64:
; CHECK-NEXT: .functype call_indirect_i64 (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @call_indirect_i64(ptr %callee) {
  %t = call i64 %callee()
  ret i64 %t
}

; CHECK-LABEL: call_indirect_float:
; CHECK-NEXT: .functype call_indirect_float (i64) -> (f32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @call_indirect_float(ptr %callee) {
  %t = call float %callee()
  ret float %t
}

; CHECK-LABEL: call_indirect_double:
; CHECK-NEXT: .functype call_indirect_double (i64) -> (f64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @call_indirect_double(ptr %callee) {
  %t = call double %callee()
  ret double %t
}

; CHECK-LABEL: call_indirect_v128:
; CHECK-NEXT: .functype call_indirect_v128 (i64) -> (v128){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $push[[NUM:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define <16 x i8> @call_indirect_v128(ptr %callee) {
  %t = call <16 x i8> %callee()
  ret <16 x i8> %t
}

; CHECK-LABEL: call_indirect_arg:
; CHECK-NEXT: .functype call_indirect_arg (i64, i32) -> (){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: {{^}} call_indirect $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return{{$}}
define void @call_indirect_arg(ptr %callee, i32 %arg) {
  call void %callee(i32 %arg)
  ret void
}

; CHECK-LABEL: call_indirect_arg_2:
; CHECK-NEXT: .functype call_indirect_arg_2 (i64, i32, i32) -> (){{$}}
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

; CHECK-LABEL: call_constexpr:
; CHECK-NEXT: .functype call_constexpr () -> (){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: call .Lvararg_func_bitcast, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i64.const	$push[[L3:[0-9]+]]=, void_nullary{{$}}
; CHECK-NEXT: i64.const	$push[[L2:[0-9]+]]=, other_void_nullary{{$}}
; CHECK-NEXT: i64.add 	$push[[L4:[0-9]+]]=, $pop[[L3]], $pop[[L2]]{{$}}
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
  call void getelementptr (i8, ptr @void_nullary, i64 ptrtoint (ptr @other_void_nullary to i64))()
  br label %bb2
bb2:
  call void inttoptr (i64 ptrtoint (ptr @void_nullary to i64) to ptr)()
  ret void
}

; Allocas should be lowered to call_indirects.
; CHECK-LABEL: call_indirect_alloca:
; CHECK:      local.tee  $push{{.*}}=, [[L0:[0-9]+]]
; CHECK-NEXT: global.set  __stack_pointer
; CHECK-NEXT: local.get  $push{{.*}}=, [[L0]]
; CHECK-NEXT: i64.const  $push{{.*}}=, 12
; CHECK-NEXT: i64.add
; CHECK-NEXT: call_indirect  $pop{{.*}}
define void @call_indirect_alloca() {
entry:
  %ptr = alloca i32, align 4
  call void %ptr()
  ret void
}

; Calling non-functional globals should be lowered to call_indirects.
; CHECK-LABEL: call_indirect_int:
; CHECK:      i64.const  $push[[L0:[0-9]+]]=, global_i8
; CHECK-NEXT: call_indirect  $pop[[L0]]
; CHECK-NEXT: i64.const  $push[[L1:[0-9]+]]=, global_i32
; CHECK-NEXT: call_indirect  $pop[[L1]]
@global_i8 = global i8 0
@global_i32 = global i32 0
define void @call_indirect_int() {
  call void @global_i8()
  call void @global_i32()
  ret void
}

; Calling aliases of non-functional globals should be lowered to call_indirects.
; CHECK-LABEL: call_indirect_int_alias:
; CHECK:      i64.const  $push[[L0:[0-9]+]]=, global_i8_alias
; CHECK-NEXT: call_indirect  $pop[[L0]]
; CHECK-NEXT: i64.const  $push[[L1:[0-9]+]]=, global_i32_alias
; CHECK-NEXT: call_indirect  $pop[[L1]]
@global_i8_alias = alias i8, ptr @global_i8
@global_i32_alias = alias i32, ptr @global_i32
define void @call_indirect_int_alias() {
  call void @global_i8_alias()
  call void @global_i32_alias()
  ret void
}

; Ideally calling aliases of functions should be lowered to direct calls. We
; support this in the normal (=slow) isel.
; CHECK-LABEL: call_func_alias:
; SLOW:      call  func_alias
; FAST:      i64.const  $push[[L0:[0-9]+]]=, func_alias
; FAST-NEXT: call_indirect  $pop[[L0]]
@func_alias = alias void (), ptr @call_void_nullary
define void @call_func_alias() {
  call void @func_alias()
  ret void
}
