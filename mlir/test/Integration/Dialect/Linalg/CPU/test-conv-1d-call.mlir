// RUN: mlir-opt %s -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf  -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:    -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

// Creates and returns a 1-D buffer of size %s1 filled with the value %f
func.func @alloc_1d_filled_f32(%s1 : index, %f : f32) -> memref<?xf32> {
  %buf = memref.alloc(%s1) : memref<?xf32>
  linalg.fill ins(%f : f32) outs(%buf : memref<?xf32>)
  return %buf : memref<?xf32>
}

func.func @conv_1d(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  linalg.conv_1d ins (%arg0, %arg1: memref<?xf32>, memref<?xf32>)
                outs (%arg2: memref<?xf32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_1d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 [4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}

func.func @main() {
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %filter1D = call @alloc_1d_filled_f32(%c3, %val) : (index, f32) -> (memref<?xf32>)
  %in1D = call @alloc_1d_filled_f32(%c8, %val) : (index, f32) -> (memref<?xf32>)
  %out1D = call @alloc_1d_filled_f32(%c6, %zero) : (index, f32) -> (memref<?xf32>)

  memref.store %f10, %in1D[%c3] : memref<?xf32>
  call @conv_1d(%in1D, %filter1D, %out1D) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  %out1D_ = memref.cast %out1D : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%out1D_): (memref<*xf32>) -> ()

  memref.dealloc %filter1D : memref<?xf32>
  memref.dealloc %in1D : memref<?xf32>
  memref.dealloc %out1D : memref<?xf32>
  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [12, 28, 28, 28, 12, 12]
