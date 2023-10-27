// RUN: mlir-opt -convert-func-to-llvm='use-opaque-pointers=1' -split-input-file -verify-diagnostics %s | FileCheck %s

//CHECK: llvm.func @second_order_arg(!llvm.ptr)
func.func private @second_order_arg(%arg0 : () -> ())

//CHECK: llvm.func @second_order_result() -> !llvm.ptr
func.func private @second_order_result() -> (() -> ())

//CHECK: llvm.func @second_order_multi_result() -> !llvm.struct<(ptr, ptr, ptr)>
func.func private @second_order_multi_result() -> (() -> (i32), () -> (i64), () -> (f32))

// Check that memrefs are converted to argument packs if appear as function arguments.
// CHECK: llvm.func @memref_call_conv(!llvm.ptr, !llvm.ptr, i64, i64, i64)
func.func private @memref_call_conv(%arg0: memref<?xf32>)

// Same in nested functions.
// CHECK: llvm.func @memref_call_conv_nested(!llvm.ptr)
func.func private @memref_call_conv_nested(%arg0: (memref<?xf32>) -> ())

//CHECK-LABEL: llvm.func @pass_through(%arg0: !llvm.ptr) -> !llvm.ptr {
func.func @pass_through(%arg0: () -> ()) -> (() -> ()) {
// CHECK-NEXT:  llvm.br ^bb1(%arg0 : !llvm.ptr)
  cf.br ^bb1(%arg0 : () -> ())

//CHECK-NEXT: ^bb1(%0: !llvm.ptr):
^bb1(%bbarg: () -> ()):
// CHECK-NEXT:  llvm.return %0 : !llvm.ptr
  return %bbarg : () -> ()
}

// CHECK-LABEL: llvm.func extern_weak @llvmlinkage(i32)
func.func private @llvmlinkage(i32) attributes { "llvm.linkage" = #llvm.linkage<extern_weak> }

// CHECK-LABEL: llvm.func @llvmreadnone(i32)
// CHECK-SAME: memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
func.func private @llvmreadnone(i32) attributes { llvm.readnone }

// CHECK-LABEL: llvm.func @body(i32)
func.func private @body(i32)

// CHECK-LABEL: llvm.func @indirect_const_call
// CHECK-SAME: (%[[ARG0:.*]]: i32) {
func.func @indirect_const_call(%arg0: i32) {
// CHECK-NEXT: %[[ADDR:.*]] = llvm.mlir.addressof @body : !llvm.ptr
  %0 = constant @body : (i32) -> ()
// CHECK-NEXT:  llvm.call %[[ADDR]](%[[ARG0:.*]]) : !llvm.ptr, (i32) -> ()
  call_indirect %0(%arg0) : (i32) -> ()
// CHECK-NEXT:  llvm.return
  return
}

// CHECK-LABEL: llvm.func @indirect_call(%arg0: !llvm.ptr, %arg1: f32) -> i32 {
func.func @indirect_call(%arg0: (f32) -> i32, %arg1: f32) -> i32 {
// CHECK-NEXT:  %0 = llvm.call %arg0(%arg1) : !llvm.ptr, (f32) -> i32
  %0 = call_indirect %arg0(%arg1) : (f32) -> i32
// CHECK-NEXT:  llvm.return %0 : i32
  return %0 : i32
}

func.func @variadic_func(%arg0: i32) attributes { "func.varargs" = true } {
  return
}

// -----

// CHECK-LABEL: llvm.func @private_callee
// CHECK-SAME: sym_visibility = "private"
func.func private @private_callee(%arg1: f32) -> i32 {
  %0 = arith.constant 0 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @caller_private_callee
func.func @caller_private_callee(%arg1: f32) -> i32 {
  %0 = call @private_callee(%arg1) : (f32) -> i32
  return %0 : i32
}

// -----

func.func private @badllvmlinkage(i32) attributes { "llvm.linkage" = 3 : i64 } // expected-error {{Contains llvm.linkage attribute not of type LLVM::LinkageAttr}}

// -----

// expected-error@+1{{C interface for variadic functions is not supported yet.}}
func.func @variadic_func(%arg0: i32) attributes { "func.varargs" = true, "llvm.emit_c_interface" } {
  return
}
