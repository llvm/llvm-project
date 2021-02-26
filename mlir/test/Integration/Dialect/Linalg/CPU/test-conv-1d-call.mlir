// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=4" -convert-linalg-to-loops -convert-scf-to-std \
// RUN:   -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-conv-vectorization="tile-sizes=1,3" -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=4" \
// RUN:   -test-conv-vectorization="tile-sizes=1,3" -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

// Creates and returns a 1-D buffer of size %s1 filled with the value %f
func @alloc_1d_filled_f32(%s1 : index, %f : f32) -> memref<?xf32> {
  %buf = alloc(%s1) : memref<?xf32>
  linalg.fill(%buf, %f) : memref<?xf32>, f32
  return %buf : memref<?xf32>
}

func @conv_1d(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  linalg.conv_1d ins (%arg0, %arg1: memref<?xf32>, memref<?xf32>)
                outs (%arg2: memref<?xf32>)
  return
}

func @main() {
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %c8 = constant 8 : index
  %f10 = constant 10.00000e+00 : f32
  %val = constant 2.00000e+00 : f32
  %zero = constant 0.00000e+00 : f32

  %filter1D = call @alloc_1d_filled_f32(%c3, %val) : (index, f32) -> (memref<?xf32>)
  %in1D = call @alloc_1d_filled_f32(%c8, %val) : (index, f32) -> (memref<?xf32>)
  %out1D = call @alloc_1d_filled_f32(%c6, %zero) : (index, f32) -> (memref<?xf32>)

  store %f10, %in1D[%c3] : memref<?xf32>
  call @conv_1d(%in1D, %filter1D, %out1D) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  %out1D_ = memref_cast %out1D : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%out1D_): (memref<*xf32>) -> ()

  dealloc %filter1D : memref<?xf32>
  dealloc %in1D : memref<?xf32>
  dealloc %out1D : memref<?xf32>
  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [12, 28, 28, 28, 12, 12]
