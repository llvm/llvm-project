// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext,%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42.0 : f32

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
  call @printMemrefF32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [0,   1,   2]
  // CHECK-NEXT: [3,   4,   5]

  %copy = memref.alloc() : memref<2x3xf32>
  memref.copy %input, %copy : memref<2x3xf32> to memref<2x3xf32>
  %unranked_copy = memref.cast %copy : memref<2x3xf32> to memref<*xf32>
  call @printMemrefF32(%unranked_copy) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [0,   1,   2]
  // CHECK-NEXT: [3,   4,   5]

  %copy_two = memref.alloc() : memref<3x2xf32>
  %copy_two_casted = memref.reinterpret_cast %copy_two to offset: [0], sizes: [2, 3], strides: [1, 2]
    : memref<3x2xf32> to memref<2x3xf32, strided<[1, 2], offset: 0>>
  memref.copy %input, %copy_two_casted : memref<2x3xf32> to memref<2x3xf32, strided<[1, 2], offset: 0>>
  %unranked_copy_two = memref.cast %copy_two : memref<3x2xf32> to memref<*xf32>
  call @printMemrefF32(%unranked_copy_two) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 2] strides = [2, 1]
  // CHECK-NEXT: [0,   3]
  // CHECK-NEXT: [1,   4]
  // CHECK-NEXT: [2,   5]

  %input_empty = memref.alloc() : memref<3x0x1xf32>
  %copy_empty = memref.alloc() : memref<3x0x1xf32>
  // Copying an empty shape should do nothing (and should not crash).
  memref.copy %input_empty, %copy_empty : memref<3x0x1xf32> to memref<3x0x1xf32>

  %input_empty_casted = memref.reinterpret_cast %input_empty to offset: [0], sizes: [0, 3, 1], strides: [3, 1, 1]
    : memref<3x0x1xf32> to memref<0x3x1xf32, strided<[3, 1, 1], offset: 0>>
  %copy_empty_casted = memref.alloc() : memref<0x3x1xf32>
  // Copying a casted empty shape should do nothing (and should not crash).
  memref.copy %input_empty_casted, %copy_empty_casted : memref<0x3x1xf32, strided<[3, 1, 1], offset: 0>> to memref<0x3x1xf32>

  %scalar = memref.alloc() : memref<f32>
  memref.store %c42, %scalar[] : memref<f32>
  %scalar_copy = memref.alloc() : memref<f32>
  memref.copy %scalar, %scalar_copy : memref<f32> to memref<f32>
  %unranked_scalar_copy = memref.cast %scalar_copy : memref<f32> to memref<*xf32>
  call @printMemrefF32(%unranked_scalar_copy) : (memref<*xf32>) -> ()
  // CHECK: rank = 0 offset = 0 sizes = [] strides = []
  // CHECK-NEXT [42]

  memref.dealloc %copy_empty : memref<3x0x1xf32>
  memref.dealloc %copy_empty_casted : memref<0x3x1xf32>
  memref.dealloc %input_empty : memref<3x0x1xf32>
  memref.dealloc %copy_two : memref<3x2xf32>
  memref.dealloc %copy : memref<2x3xf32>
  memref.dealloc %input : memref<2x3xf32>
  memref.dealloc %scalar : memref<f32>
  memref.dealloc %scalar_copy : memref<f32>
  return
}
