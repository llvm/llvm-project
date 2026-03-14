// RUN: mlir-opt %s -buffer-deallocation-pipeline | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(buffer-deallocation-pipeline)" | FileCheck %s

// Test case requires a helper function.

// CHECK-LABEL: func.func @test(
//       CHECK:   func.call @[[helper:.*]]({{.*}}) : ({{.*}}) -> ()
func.func @test(%lb : index, %ub: index) -> (memref<5xf32>, memref<5xf32>) {
  %0 = memref.alloc() : memref<5xf32>
  %1 = memref.alloc() : memref<5xf32>
  %c1 = arith.constant 1 : index
  %a, %b = scf.for %iv = %lb to %ub step %c1 iter_args(%c = %0, %d = %1)
      -> (memref<5xf32>, memref<5xf32>) {
    scf.yield %d, %c : memref<5xf32>, memref<5xf32>
  }
  return %a, %b : memref<5xf32>, memref<5xf32>
}

//      CHECK: func.func private @[[helper]](
// CHECK-SAME:     %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xi1>, %{{.*}}: memref<?xi1>, %{{.*}}: memref<?xi1>)
