// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-linalg-to-loops,convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" |        \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

// CHECK: rank = 2
// CHECK-SAME: sizes = [10, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-10: [10, 10, 10]
//
// CHECK: rank = 2
// CHECK-SAME: sizes = [10, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-10: [5, 5, 5]
//
// CHECK: rank = 2
// CHECK-SAME: sizes = [10, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-10: [2, 2, 2]
//
// CHECK: rank = 0
// 122 is ASCII for 'z'.
// CHECK: [z]
//
// CHECK: rank = 2
// CHECK-SAME: sizes = [4, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-4: [1, 1, 1]
//
// CHECK: rank = 2
// CHECK-SAME: sizes = [4, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-4: [1, 1, 1]
//
// CHECK: rank = 2
// CHECK-SAME: sizes = [4, 3]
// CHECK-SAME: strides = [3, 1]
// CHECK-COUNT-4: [1, 1, 1]
func.func @main() -> () {
    %A = memref.alloc() : memref<10x3xf32, 0>
    %f2 = arith.constant 2.00000e+00 : f32
    %f5 = arith.constant 5.00000e+00 : f32
    %f10 = arith.constant 10.00000e+00 : f32

    %V = memref.cast %A : memref<10x3xf32, 0> to memref<?x?xf32>
    linalg.fill ins(%f10 : f32) outs(%V : memref<?x?xf32, 0>)
    %U = memref.cast %A : memref<10x3xf32, 0> to memref<*xf32>
    call @printMemrefF32(%U) : (memref<*xf32>) -> ()

    %V2 = memref.cast %U : memref<*xf32> to memref<?x?xf32>
    linalg.fill ins(%f5 : f32) outs(%V2 : memref<?x?xf32, 0>)
    %U2 = memref.cast %V2 : memref<?x?xf32, 0> to memref<*xf32>
    call @printMemrefF32(%U2) : (memref<*xf32>) -> ()

    %V3 = memref.cast %V2 : memref<?x?xf32> to memref<*xf32>
    %V4 = memref.cast %V3 : memref<*xf32> to memref<?x?xf32>
    linalg.fill ins(%f2 : f32) outs(%V4 : memref<?x?xf32, 0>)
    %U3 = memref.cast %V2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%U3) : (memref<*xf32>) -> ()

    // 122 is ASCII for 'z'.
    %i8_z = arith.constant 122 : i8
    %I8 = memref.alloc() : memref<i8>
    memref.store %i8_z, %I8[]: memref<i8>
    %U4 = memref.cast %I8 : memref<i8> to memref<*xi8>
    call @printMemrefI8(%U4) : (memref<*xi8>) -> ()

    memref.dealloc %U4 : memref<*xi8>
    memref.dealloc %A : memref<10x3xf32, 0>

    call @return_var_memref_caller() : () -> ()
    call @return_two_var_memref_caller() : () -> ()
    call @dim_op_of_unranked() : () -> ()
    return
}

func.func private @printMemrefI8(memref<*xi8>) attributes { llvm.emit_c_interface }
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @return_two_var_memref_caller() {
  %0 = memref.alloca() : memref<4x3xf32>
  %c0f32 = arith.constant 1.0 : f32
  linalg.fill ins(%c0f32 : f32) outs(%0 : memref<4x3xf32>)
  %1:2 = call @return_two_var_memref(%0) : (memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>)
  call @printMemrefF32(%1#0) : (memref<*xf32>) -> ()
  call @printMemrefF32(%1#1) : (memref<*xf32>) -> ()
  return
 }

 func.func @return_two_var_memref(%arg0: memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>) {
  %0 = memref.cast %arg0 : memref<4x3xf32> to memref<*xf32>
  return %0, %0 : memref<*xf32>, memref<*xf32>
}

func.func @return_var_memref_caller() {
  %0 = memref.alloca() : memref<4x3xf32>
  %c0f32 = arith.constant 1.0 : f32
  linalg.fill ins(%c0f32 : f32) outs(%0 : memref<4x3xf32>)
  %1 = call @return_var_memref(%0) : (memref<4x3xf32>) -> memref<*xf32>
  call @printMemrefF32(%1) : (memref<*xf32>) -> ()
  return
}

func.func @return_var_memref(%arg0: memref<4x3xf32>) -> memref<*xf32> {
  %0 = memref.cast %arg0: memref<4x3xf32> to memref<*xf32>
  return %0 : memref<*xf32>
}

func.func private @printU64(index) -> ()
func.func private @printNewline() -> ()

func.func @dim_op_of_unranked() {
  %ranked = memref.alloca() : memref<4x3xf32>
  %unranked = memref.cast %ranked: memref<4x3xf32> to memref<*xf32>

  %c0 = arith.constant 0 : index
  %dim_0 = memref.dim %unranked, %c0 : memref<*xf32>
  call @printU64(%dim_0) : (index) -> ()
  call @printNewline() : () -> ()
  // CHECK: 4

  %c1 = arith.constant 1 : index
  %dim_1 = memref.dim %unranked, %c1 : memref<*xf32>
  call @printU64(%dim_1) : (index) -> ()
  call @printNewline() : () -> ()
  // CHECK: 3

  return
}
