// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s

//CHECK: llvm.func @second_order_arg(!llvm<"void ()*">)
func @second_order_arg(%arg0 : () -> ())

//CHECK: llvm.func @second_order_result() -> !llvm<"void ()*">
func @second_order_result() -> (() -> ())

//CHECK: llvm.func @second_order_multi_result() -> !llvm<"{ i32 ()*, i64 ()*, float ()* }">
func @second_order_multi_result() -> (() -> (i32), () -> (i64), () -> (f32))

//CHECK: llvm.func @third_order(!llvm<"void ()* (void ()*)*">) -> !llvm<"void ()* (void ()*)*">
func @third_order(%arg0 : (() -> ()) -> (() -> ())) -> ((() -> ()) -> (() -> ()))

//CHECK: llvm.func @fifth_order_left(!llvm<"void (void (void (void ()*)*)*)*">)
func @fifth_order_left(%arg0: (((() -> ()) -> ()) -> ()) -> ())

//CHECK: llvm.func @fifth_order_right(!llvm<"void ()* ()* ()* ()*">)
func @fifth_order_right(%arg0: () -> (() -> (() -> (() -> ()))))

// Check that memrefs are converted to argument packs if appear as function arguments.
// CHECK: llvm.func @memref_call_conv(!llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64)
func @memref_call_conv(%arg0: memref<?xf32>)

// Same in nested functions.
// CHECK: llvm.func @memref_call_conv_nested(!llvm<"void (float*, float*, i64, i64, i64)*">)
func @memref_call_conv_nested(%arg0: (memref<?xf32>) -> ())

//CHECK-LABEL: llvm.func @pass_through(%arg0: !llvm<"void ()*">) -> !llvm<"void ()*"> {
func @pass_through(%arg0: () -> ()) -> (() -> ()) {
// CHECK-NEXT:  llvm.br ^bb1(%arg0 : !llvm<"void ()*">)
  br ^bb1(%arg0 : () -> ())

//CHECK-NEXT: ^bb1(%0: !llvm<"void ()*">):
^bb1(%bbarg: () -> ()):
// CHECK-NEXT:  llvm.return %0 : !llvm<"void ()*">
  return %bbarg : () -> ()
}

// CHECK-LABEL: llvm.func @body(!llvm.i32)
func @body(i32)

// CHECK-LABEL: llvm.func @indirect_const_call
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.i32) {
func @indirect_const_call(%arg0: i32) {
// CHECK-NEXT: %[[ADDR:.*]] = llvm.mlir.addressof @body : !llvm<"void (i32)*">
  %0 = constant @body : (i32) -> ()
// CHECK-NEXT:  llvm.call %[[ADDR]](%[[ARG0:.*]]) : (!llvm.i32) -> ()
  call_indirect %0(%arg0) : (i32) -> ()
// CHECK-NEXT:  llvm.return
  return
}

// CHECK-LABEL: llvm.func @indirect_call(%arg0: !llvm<"i32 (float)*">, %arg1: !llvm.float) -> !llvm.i32 {
func @indirect_call(%arg0: (f32) -> i32, %arg1: f32) -> i32 {
// CHECK-NEXT:  %0 = llvm.call %arg0(%arg1) : (!llvm.float) -> !llvm.i32
  %0 = call_indirect %arg0(%arg1) : (f32) -> i32
// CHECK-NEXT:  llvm.return %0 : !llvm.i32
  return %0 : i32
}

