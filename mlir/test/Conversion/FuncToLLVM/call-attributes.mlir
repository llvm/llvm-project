// RUN: mlir-opt -convert-func-to-llvm %s | FileCheck %s

func.func private @callee(f32) -> f32

// CHECK-LABEL: llvm.func @direct_call
func.func @direct_call(%arg: f32) -> f32 {
  // CHECK: llvm.call fastcc @callee(%arg0) {convergent, fastmathFlags = #llvm.fastmath<fast>, test.marker = "keep"}
  %0 = func.call @callee(%arg) {CConv = #llvm.cconv<fastcc>, fastmathFlags = #llvm.fastmath<fast>, convergent, test.marker = "keep"} : (f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: llvm.func @indirect_call
func.func @indirect_call(%fn: (f32) -> f32, %arg: f32) -> f32 {
  // CHECK: llvm.call fastcc %arg0(%arg1) {convergent, fastmathFlags = #llvm.fastmath<nnan>, test.marker = "keep"}
  %0 = func.call_indirect %fn(%arg) {CConv = #llvm.cconv<fastcc>, fastmathFlags = #llvm.fastmath<nnan>, convergent, test.marker = "keep"} : (f32) -> f32
  return %0 : f32
}

// The `llvm.` argument and result attributes and `no_inline` are forwarded
// when operands and results map one-to-one.

func.func private @callee_i16(i16) -> i16
func.func private @callee_memref(memref<4xf32>)
func.func private @callee_two() -> (i16, i16)

// CHECK-LABEL: llvm.func @call_arg_res_attrs
func.func @call_arg_res_attrs(%v: i16) -> i16 {
  // CHECK: llvm.call @callee_i16(%arg0) {no_inline} : (i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext})
  %r = "func.call"(%v) <{callee = @callee_i16, arg_attrs = [{llvm.noundef, llvm.signext}], res_attrs = [{llvm.signext}], no_inline}> : (i16) -> i16
  return %r : i16
}

// CHECK-LABEL: llvm.func @call_indirect_arg_res_attrs
func.func @call_indirect_arg_res_attrs(%fn: (i16) -> i16, %v: i16) -> i16 {
  // CHECK: llvm.call %arg0(%arg1) : !llvm.ptr, (i16 {llvm.signext}) -> (i16 {llvm.zeroext})
  %r = "func.call_indirect"(%fn, %v) <{arg_attrs = [{llvm.signext}], res_attrs = [{llvm.zeroext}]}> : ((i16) -> i16, i16) -> i16
  return %r : i16
}

// Attributes from other dialects are not forwarded.
// CHECK-LABEL: llvm.func @call_non_llvm_attrs
func.func @call_non_llvm_attrs(%v: i16) -> i16 {
  // CHECK: llvm.call @callee_i16(%arg0) : (i16) -> i16
  %r = "func.call"(%v) <{callee = @callee_i16, arg_attrs = [{test.marker}], res_attrs = [{test.marker}]}> : (i16) -> i16
  return %r : i16
}

// A memref operand expands to several values: no positional mapping.
// CHECK-LABEL: llvm.func @call_expanded_operand
func.func @call_expanded_operand(%m: memref<4xf32>) {
  // CHECK: llvm.call @callee_memref({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
  "func.call"(%m) <{callee = @callee_memref, arg_attrs = [{llvm.noundef}]}> : (memref<4xf32>) -> ()
  return
}

// Several results are packed into one struct: no positional mapping.
// CHECK-LABEL: llvm.func @call_packed_results
func.func @call_packed_results() -> (i16, i16) {
  // CHECK: llvm.call @callee_two() : () -> !llvm.struct<(i16, i16)>
  %a, %b = "func.call"() <{callee = @callee_two, res_attrs = [{llvm.signext}, {llvm.zeroext}]}> : () -> (i16, i16)
  return %a, %b : i16, i16
}
