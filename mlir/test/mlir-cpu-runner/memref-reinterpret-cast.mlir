// RUN: mlir-opt %s -pass-pipeline="func.func(convert-scf-to-cf),convert-memref-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts" \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Initialize input.
  %input = memref.alloc() : memref<2x3xf32>
  %dim_x = memref.dim %input, %c0 : memref<2x3xf32>
  %dim_y = memref.dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %prod = arith.muli %i,  %dim_y : index
    %val = arith.addi %prod, %j : index
    %val_i64 = arith.index_cast %val : index to i64
    %val_f32 = arith.sitofp %val_i64 : i64 to f32
    memref.store %val_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [0,   1,   2]
  // CHECK-NEXT: [3,   4,   5]

  // Test cases.
  call @cast_ranked_memref_to_static_shape(%input) : (memref<2x3xf32>) -> ()
  call @cast_ranked_memref_to_dynamic_shape(%input) : (memref<2x3xf32>) -> ()
  call @cast_unranked_memref_to_static_shape(%input) : (memref<2x3xf32>) -> ()
  call @cast_unranked_memref_to_dynamic_shape(%input) : (memref<2x3xf32>) -> ()
  memref.dealloc %input : memref<2x3xf32>
  return
}

func @cast_ranked_memref_to_static_shape(%input : memref<2x3xf32>) {
  %output = memref.reinterpret_cast %input to
           offset: [0], sizes: [6, 1], strides: [1, 1]
           : memref<2x3xf32> to memref<6x1xf32>

  %unranked_output = memref.cast %output
      : memref<6x1xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [6, 1] strides = [1, 1] data =
  // CHECK-NEXT: [0],
  // CHECK-NEXT: [1],
  // CHECK-NEXT: [2],
  // CHECK-NEXT: [3],
  // CHECK-NEXT: [4],
  // CHECK-NEXT: [5]
  return
}

func @cast_ranked_memref_to_dynamic_shape(%input : memref<2x3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %output = memref.reinterpret_cast %input to
           offset: [%c0], sizes: [%c1, %c6], strides: [%c6, %c1]
           : memref<2x3xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>

  %unranked_output = memref.cast %output
      : memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [1, 6] strides = [6, 1] data =
  // CHECK-NEXT: [0,   1,   2,   3,   4,   5]
  return
}

func @cast_unranked_memref_to_static_shape(%input : memref<2x3xf32>) {
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  %output = memref.reinterpret_cast %unranked_input to
           offset: [0], sizes: [6, 1], strides: [1, 1]
           : memref<*xf32> to memref<6x1xf32>

  %unranked_output = memref.cast %output
      : memref<6x1xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [6, 1] strides = [1, 1] data =
  // CHECK-NEXT: [0],
  // CHECK-NEXT: [1],
  // CHECK-NEXT: [2],
  // CHECK-NEXT: [3],
  // CHECK-NEXT: [4],
  // CHECK-NEXT: [5]
  return
}

func @cast_unranked_memref_to_dynamic_shape(%input : memref<2x3xf32>) {
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %output = memref.reinterpret_cast %unranked_input to
           offset: [%c0], sizes: [%c1, %c6], strides: [%c6, %c1]
           : memref<*xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>

  %unranked_output = memref.cast %output
      : memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [1, 6] strides = [6, 1] data =
  // CHECK-NEXT: [0,   1,   2,   3,   4,   5]
  return
}
