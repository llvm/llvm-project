// RUN: mlir-opt %s -convert-linalg-to-loops \
// RUN:             -convert-scf-to-std      \
// RUN:             -convert-linalg-to-llvm  \
// RUN:             -convert-std-to-llvm |   \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

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
func @main() -> () {
    %A = memref.alloc() : memref<10x3xf32, 0>
    %f2 = constant 2.00000e+00 : f32
    %f5 = constant 5.00000e+00 : f32
    %f10 = constant 10.00000e+00 : f32

    %V = memref.cast %A : memref<10x3xf32, 0> to memref<?x?xf32>
    linalg.fill(%V, %f10) : memref<?x?xf32, 0>, f32
    %U = memref.cast %A : memref<10x3xf32, 0> to memref<*xf32>
    call @print_memref_f32(%U) : (memref<*xf32>) -> ()

    %V2 = memref.cast %U : memref<*xf32> to memref<?x?xf32>
    linalg.fill(%V2, %f5) : memref<?x?xf32, 0>, f32
    %U2 = memref.cast %V2 : memref<?x?xf32, 0> to memref<*xf32>
    call @print_memref_f32(%U2) : (memref<*xf32>) -> ()

    %V3 = memref.cast %V2 : memref<?x?xf32> to memref<*xf32>
    %V4 = memref.cast %V3 : memref<*xf32> to memref<?x?xf32>
    linalg.fill(%V4, %f2) : memref<?x?xf32, 0>, f32
    %U3 = memref.cast %V2 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%U3) : (memref<*xf32>) -> ()

    // 122 is ASCII for 'z'.
    %i8_z = constant 122 : i8
    %I8 = memref.alloc() : memref<i8>
    memref.store %i8_z, %I8[]: memref<i8>
    %U4 = memref.cast %I8 : memref<i8> to memref<*xi8>
    call @print_memref_i8(%U4) : (memref<*xi8>) -> ()

    memref.dealloc %A : memref<10x3xf32, 0>

    call @return_var_memref_caller() : () -> ()
    call @return_two_var_memref_caller() : () -> ()
    call @dim_op_of_unranked() : () -> ()
    return
}

func private @print_memref_i8(memref<*xi8>) attributes { llvm.emit_c_interface }
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @return_two_var_memref_caller() {
  %0 = memref.alloca() : memref<4x3xf32>
  %c0f32 = constant 1.0 : f32
  linalg.fill(%0, %c0f32) : memref<4x3xf32>, f32
  %1:2 = call @return_two_var_memref(%0) : (memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>)
  call @print_memref_f32(%1#0) : (memref<*xf32>) -> ()
  call @print_memref_f32(%1#1) : (memref<*xf32>) -> ()
  return
 }

 func @return_two_var_memref(%arg0: memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>) {
  %0 = memref.cast %arg0 : memref<4x3xf32> to memref<*xf32>
  return %0, %0 : memref<*xf32>, memref<*xf32>
}

func @return_var_memref_caller() {
  %0 = memref.alloca() : memref<4x3xf32>
  %c0f32 = constant 1.0 : f32
  linalg.fill(%0, %c0f32) : memref<4x3xf32>, f32
  %1 = call @return_var_memref(%0) : (memref<4x3xf32>) -> memref<*xf32>
  call @print_memref_f32(%1) : (memref<*xf32>) -> ()
  return
}

func @return_var_memref(%arg0: memref<4x3xf32>) -> memref<*xf32> {
  %0 = memref.cast %arg0: memref<4x3xf32> to memref<*xf32>
  return %0 : memref<*xf32>
}

func private @printU64(index) -> ()
func private @printNewline() -> ()

func @dim_op_of_unranked() {
  %ranked = memref.alloc() : memref<4x3xf32>
  %unranked = memref.cast %ranked: memref<4x3xf32> to memref<*xf32>

  %c0 = constant 0 : index
  %dim_0 = dim %unranked, %c0 : memref<*xf32>
  call @printU64(%dim_0) : (index) -> ()
  call @printNewline() : () -> ()
  // CHECK: 4

  %c1 = constant 1 : index
  %dim_1 = dim %unranked, %c1 : memref<*xf32>
  call @printU64(%dim_1) : (index) -> ()
  call @printNewline() : () -> ()
  // CHECK: 3

  return
}
