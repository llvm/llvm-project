// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,2" -convert-linalg-to-loops -convert-scf-to-std \
// RUN:   -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-conv-vectorization="tile-sizes=1,1,3,3"  -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,2" \
// RUN:   -test-conv-vectorization="tile-sizes=1,1,3,3"  -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

// Creates and returns a 2-D buffer of size (%s1, %s2) filled with the value %f
func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill(%f, %buf) : f32, memref<?x?xf32>
  return %buf : memref<?x?xf32>
}

func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                outs (%arg2: memref<?x?xf32>)
  return
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %c8 = constant 8 : index
  %f10 = constant 10.00000e+00 : f32
  %val = constant 2.00000e+00 : f32
  %zero = constant 0.00000e+00 : f32

  %filter2D = call @alloc_2d_filled_f32(%c3, %c3, %val) : (index, index, f32) -> (memref<?x?xf32>)
  %in2D = call @alloc_2d_filled_f32(%c8, %c8, %val) : (index, index, f32) -> (memref<?x?xf32>)
  %out2D = call @alloc_2d_filled_f32(%c6, %c6, %zero) : (index, index, f32) -> (memref<?x?xf32>)

  memref.store %f10, %in2D[%c0, %c3] : memref<?x?xf32>
  call @conv_2d(%in2D, %filter2D, %out2D) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %out2D_ = memref.cast %out2D : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%out2D_): (memref<*xf32>) -> ()

  memref.dealloc %filter2D : memref<?x?xf32>
  memref.dealloc %in2D : memref<?x?xf32>
  memref.dealloc %out2D : memref<?x?xf32>
  return
}

// CHECK:        Unranked Memref {{.*}}
// CHECK-NEXT:   [
// CHECK-SAME:    [36,   52,   52,   52,   36,   36],
// CHECK-COUNT-5: [36,   36,   36,   36,   36,   36]
// CHECK-SAME:   ]
