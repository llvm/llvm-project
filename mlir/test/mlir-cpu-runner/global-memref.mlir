// RUN: mlir-opt %s -pass-pipeline="builtin.func(convert-arith-to-llvm),convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts" | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_i32(memref<*xi32>) attributes { llvm.emit_c_interface }
func private @printNewline() -> ()

memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
func @test1DMemref() {
  %0 = memref.get_global @gv0 : memref<4xf32>
  %U = memref.cast %0 : memref<4xf32> to memref<*xf32>
  // CHECK: rank = 1
  // CHECK: offset = 0
  // CHECK: sizes = [4]
  // CHECK: strides = [1]
  // CHECK: [0,  1,  2,  3]
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  call @printNewline() : () -> ()

  // Overwrite some of the elements.
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %fp0 = arith.constant 4.0 : f32
  %fp1 = arith.constant 5.0 : f32
  memref.store %fp0, %0[%c0] : memref<4xf32>
  memref.store %fp1, %0[%c2] : memref<4xf32>
  // CHECK: rank = 1
  // CHECK: offset = 0
  // CHECK: sizes = [4]
  // CHECK: strides = [1]
  // CHECK: [4,  1,  5,  3]
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  call @printNewline() : () -> ()
  return
}

memref.global constant @gv1 : memref<3x2xi32> = dense<[[0, 1],[2, 3],[4, 5]]>
func @testConstantMemref() {
  %0 = memref.get_global @gv1 : memref<3x2xi32>
  %U = memref.cast %0 : memref<3x2xi32> to memref<*xi32>
  // CHECK: rank = 2
  // CHECK: offset = 0
  // CHECK: sizes = [3, 2]
  // CHECK: strides = [2, 1]
  // CHECK: [0,   1]
  // CHECK: [2,   3]
  // CHECK: [4,   5]
  call @print_memref_i32(%U) : (memref<*xi32>) -> ()
  call @printNewline() : () -> ()
  return
}

memref.global "private" @gv2 : memref<4x2xf32> = dense<[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]>
func @test2DMemref() {
  %0 = memref.get_global @gv2 : memref<4x2xf32>
  %U = memref.cast %0 : memref<4x2xf32> to memref<*xf32>
  // CHECK: rank = 2
  // CHECK: offset = 0
  // CHECK: sizes = [4, 2]
  // CHECK: strides = [2, 1]
  // CHECK: [0,   1]
  // CHECK: [2,   3]
  // CHECK: [4,   5]
  // CHECK: [6,   7]
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  call @printNewline() : () -> ()

  // Overwrite the 1.0 (at index [0, 1]) with 10.0
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %fp10 = arith.constant 10.0 : f32
  memref.store %fp10, %0[%c0, %c1] : memref<4x2xf32>
  // CHECK: rank = 2
  // CHECK: offset = 0
  // CHECK: sizes = [4, 2]
  // CHECK: strides = [2, 1]
  // CHECK: [0,   10]
  // CHECK: [2,   3]
  // CHECK: [4,   5]
  // CHECK: [6,   7]
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  call @printNewline() : () -> ()
  return
}

memref.global @gv3 : memref<i32> = dense<11>
func @testScalarMemref() {
  %0 = memref.get_global @gv3 : memref<i32>
  %U = memref.cast %0 : memref<i32> to memref<*xi32>
  // CHECK: rank = 0
  // CHECK: offset = 0
  // CHECK: sizes = []
  // CHECK: strides = []
  // CHECK: [11]
  call @print_memref_i32(%U) : (memref<*xi32>) -> ()
  call @printNewline() : () -> ()
  return
}

func @main() -> () {
  call @test1DMemref() : () -> ()
  call @testConstantMemref() : () -> ()
  call @test2DMemref() : () -> ()
  call @testScalarMemref() : () -> ()
  return
}


