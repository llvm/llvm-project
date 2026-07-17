; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; ct.select's value type is restricted to single-value types: aggregates are
; accepted by the intrinsic signature (llvm_any_ty) but rejected here because
; codegen does not support them.

declare {i32, i32} @llvm.ct.select.sl_i32i32s(i1, {i32, i32}, {i32, i32})
declare [2 x i64] @llvm.ct.select.a2i64(i1, [2 x i64], [2 x i64])

; CHECK: llvm.ct.select only supports integer, floating-point, or pointer types, or vectors of them
define {i32, i32} @ct_select_struct(i1 %c, {i32, i32} %x, {i32, i32} %y) {
  %r = call {i32, i32} @llvm.ct.select.sl_i32i32s(i1 %c, {i32, i32} %x, {i32, i32} %y)
  ret {i32, i32} %r
}

; CHECK: llvm.ct.select only supports integer, floating-point, or pointer types, or vectors of them
define [2 x i64] @ct_select_array(i1 %c, [2 x i64] %x, [2 x i64] %y) {
  %r = call [2 x i64] @llvm.ct.select.a2i64(i1 %c, [2 x i64] %x, [2 x i64] %y)
  ret [2 x i64] %r
}
