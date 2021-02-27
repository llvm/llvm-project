// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,0,4,4" -convert-linalg-to-loops -convert-scf-to-std \
// RUN:   -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-conv-vectorization="tile-sizes=1,1,1,1,3,3,3"  -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=2,0,4,4" \
// RUN:   -test-conv-vectorization="tile-sizes=1,1,1,1,3,3,3" -convert-linalg-to-llvm -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

// Creates and returns 4-D buffer of size (%s1, %s2, %s3, %s4) filled with the value %f
func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> memref<?x?x?x?xf32> {
  %buf = alloc(%s1, %s2, %s3, %s4) : memref<?x?x?x?xf32>
  linalg.fill(%buf, %f) : memref<?x?x?x?xf32>, f32
  return %buf : memref<?x?x?x?xf32>
}

func @conv_2d_input_nchw_filter_hwcf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_input_nchw_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
                          strides = dense<1> : tensor<2xi64>}
     ins (%arg0, %arg1: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
    outs (%arg2: memref<?x?x?x?xf32>)
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

  %filter2D_nchw = call @alloc_4d_filled_f32(%c3, %c3, %c1, %c1, %val) : (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)
  %in2D_nchw = call @alloc_4d_filled_f32(%c3, %c1, %c8, %c8, %val) : (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)
  %out2D_nchw = call @alloc_4d_filled_f32(%c3, %c1, %c6, %c6, %zero) : (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  store %f10, %in2D_nchw[%c0, %c0, %c0, %c3] : memref<?x?x?x?xf32>
  call @conv_2d_input_nchw_filter_hwcf(%in2D_nchw, %filter2D_nchw, %out2D_nchw) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
  %out2D_nchw_ = memref_cast %out2D_nchw : memref<?x?x?x?xf32> to memref<*xf32>
  call @print_memref_f32(%out2D_nchw_): (memref<*xf32>) -> ()

  dealloc %filter2D_nchw : memref<?x?x?x?xf32>
  dealloc %in2D_nchw : memref<?x?x?x?xf32>
  dealloc %out2D_nchw : memref<?x?x?x?xf32>
  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [
// CHECK-SAME:   [
// CHECK-SAME:    [
// CHECK-SAME:     [36,     52,     52,     52,     36,     36],
// CHECK-COUNT-5:  [36,     36,     36,     36,     36,     36]
// CHECK-SAME:    ]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-SAME:    [
// CHECK-COUNT-6:  [36,     36,     36,     36,     36,     36]
// CHECK-SAME:    ]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-SAME:    [
// CHECK-COUNT-6:  [36,     36,     36,     36,     36,     36]
// CHECK-SAME:    ]
// CHECK-SAME:   ]
// CHECK-SAME:  ]
