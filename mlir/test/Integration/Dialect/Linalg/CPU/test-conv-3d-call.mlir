// RUN: mlir-opt %s -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf --convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:   -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf --convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

// Creates and returns 3-D buffer of size (%s1, %s2, %s3) filled with the value %f
func.func @alloc_3d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %f : f32) -> memref<?x?x?xf32> {
  %buf = memref.alloc(%s1, %s2, %s3) : memref<?x?x?xf32>
  linalg.fill ins(%f : f32) outs(%buf : memref<?x?x?xf32>)
  return %buf : memref<?x?x?xf32>
}

func.func @conv_3d(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_3d ins (%arg0, %arg1: memref<?x?x?xf32>, memref<?x?x?xf32>)
                outs (%arg2: memref<?x?x?xf32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_3d"]} in %arg1
    %1, %loops:3 = transform.structured.tile %0 [2, 2, 2]
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

  %filter3D = call @alloc_3d_filled_f32(%c3, %c3, %c3, %val) : (index, index, index, f32) -> (memref<?x?x?xf32>)
  %in3D = call @alloc_3d_filled_f32(%c8, %c8, %c8, %val) : (index, index, index, f32) -> (memref<?x?x?xf32>)
  %out3D = call @alloc_3d_filled_f32(%c6, %c6, %c6, %zero) : (index, index, index, f32) -> (memref<?x?x?xf32>)

  memref.store %f10, %in3D[%c0, %c0, %c3] : memref<?x?x?xf32>
  call @conv_3d(%in3D, %filter3D, %out3D) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  %out3D_ = memref.cast %out3D : memref<?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%out3D_): (memref<*xf32>) -> ()

  memref.dealloc %filter3D : memref<?x?x?xf32>
  memref.dealloc %in3D : memref<?x?x?xf32>
  memref.dealloc %out3D : memref<?x?x?xf32>
  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [
// CHECK-SAME:   [
// CHECK-SAME:    [108,    124,    124,    124,    108,    108],
// CHECK-COUNT-5: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-6: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-6: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-6: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-6: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-6: [108,    108,    108,    108,    108,    108]
// CHECK-SAME:   ]
// CHECK-SAME:  ]
