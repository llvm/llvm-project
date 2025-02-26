// RUN: mlir-opt %s -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf  -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:    -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

// Creates and returns 3-D buffer of size (%s1, %s2, %s3) filled with the value %f
func.func @alloc_3d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %f : f32) -> memref<?x?x?xf32> {
  %buf = memref.alloc(%s1, %s2, %s3) : memref<?x?x?xf32>
  linalg.fill ins(%f : f32) outs(%buf : memref<?x?x?xf32>)
  return %buf : memref<?x?x?xf32>
}

func.func @conv_1d_nwc_wcf(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                         strides = dense<1> : tensor<1xi64>}
     ins (%arg0, %arg1: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%arg2: memref<?x?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_1d_nwc_wcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %filter1D_nwc = call @alloc_3d_filled_f32(%c3, %c1, %c1, %val) : (index, index, index, f32) -> (memref<?x?x?xf32>)
  %in1D_nwc = call @alloc_3d_filled_f32(%c3, %c8, %c1, %val) : (index, index, index, f32) -> (memref<?x?x?xf32>)
  %out1D_nwc = call @alloc_3d_filled_f32(%c3, %c6, %c1, %zero) : (index, index, index, f32) -> (memref<?x?x?xf32>)

  memref.store %f10, %in1D_nwc[%c0, %c3, %c0] : memref<?x?x?xf32>
  call @conv_1d_nwc_wcf(%in1D_nwc, %filter1D_nwc, %out1D_nwc) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  %out1D_nwc_ = memref.cast %out1D_nwc : memref<?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%out1D_nwc_): (memref<*xf32>) -> ()

  memref.dealloc %filter1D_nwc : memref<?x?x?xf32>
  memref.dealloc %in1D_nwc : memref<?x?x?xf32>
  memref.dealloc %out1D_nwc : memref<?x?x?xf32>
  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [
// CHECK-SAME:   [
// CHECK-SAME:    [12],
// CHECK-COUNT-3: [28],
// CHECK-NEXT:    [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-5: [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-5: [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ]
// CHECK-SAME:  ]
