// RUN: mlir-opt %s -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,3,4" -linalg-promote-subviews -convert-linalg-to-loops -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,3,4" -linalg-promote-subviews -convert-linalg-to-std -convert-linalg-to-llvm \
// RUN: | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libmlir_test_cblas%shlibext,%linalg_test_lib_dir/libmlir_test_cblas_interface%shlibext \
// RUN: | FileCheck %s

// Creates and returns a 1-D buffer of size %s filled with the value %f
func @alloc_filled_f32(%s : index, %f : f32) -> memref<?xi8> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %s4 = muli %s, %c4: index
  %buf = alloc(%s4) {alignment = 256} : memref<?xi8>
  %V = view %buf[%c0][%s] : memref<?xi8> to memref<?xf32>
  linalg.fill(%V, %f) : memref<?xf32>, f32
  return %buf : memref<?xi8>
}

// Test for linalg.dot.
func @dot() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %f10 = constant 10.00000e+00 : f32
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c16, %f2) : (index, f32) -> (memref<?xi8>)
  %bB = call @alloc_filled_f32(%c16, %f1) : (index, f32) -> (memref<?xi8>)
  %bC = call @alloc_filled_f32(%c1, %f10) : (index, f32) -> (memref<?xi8>)

  %A = view %bA[%c0][%c16] : memref<?xi8> to memref<?xf32>
  %B = view %bB[%c0][%c16] : memref<?xi8> to memref<?xf32>
  %C = view %bC[%c0][] : memref<?xi8> to memref<f32>

  linalg.dot %A, %B, %C : (memref<?xf32>, memref<?xf32>, memref<f32>)
  %res = load %C[] : memref<f32>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}

// Test for linalg.matmul.
func @matmul() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c2 = constant 2 : index
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %c32 = constant 32 : index
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32
  %f10 = constant 10.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c32, %f2) : (index, f32) -> (memref<?xi8>)
  %bB = call @alloc_filled_f32(%c32, %f1) : (index, f32) -> (memref<?xi8>)
  %bC = call @alloc_filled_f32(%c4, %f10) : (index, f32) -> (memref<?xi8>)

  %A = view %bA[%c0][%c2, %c16] : memref<?xi8> to memref<?x?xf32>
  %B = view %bB[%c0][%c16, %c2] : memref<?xi8> to memref<?x?xf32>
  %C = view %bC[%c0][%c2, %c2] : memref<?xi8> to memref<?x?xf32>

  linalg.matmul %A, %B, %C : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
  %res = load %C[%c0, %c1] : memref<?x?xf32>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}

// All tests return this value
// CHECK: 4.2{{0+}}e+01
