// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(buffer-deallocation-pipeline))" -verify-diagnostics

// Test case requires a helper function but is run on a function.

// CHECK-LABEL: func.func @test(
//       CHECK:   func.call @[[helper:.*]]({{.*}}) : ({{.*}}) -> ()
func.func @test(%lb : index, %ub: index) -> (memref<5xf32>, memref<5xf32>) {
  %0 = memref.alloc() : memref<5xf32>
  %1 = memref.alloc() : memref<5xf32>
  %c1 = arith.constant 1 : index
  %a, %b = scf.for %iv = %lb to %ub step %c1 iter_args(%c = %0, %d = %1)
      -> (memref<5xf32>, memref<5xf32>) {
    // expected-error @below{{library function required for generic lowering, but cannot be automatically inserted when operating on functions}}
    // expected-error @below{{failed to legalize operation 'bufferization.dealloc' that was explicitly marked illegal}}
    scf.yield %d, %c : memref<5xf32>, memref<5xf32>
  }
  return %a, %b : memref<5xf32>, memref<5xf32>
}

//      CHECK: func.func @[[helper]](
// CHECK-SAME:     %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xi1>, %{{.*}}: memref<?xi1>, %{{.*}}: memref<?xi1>)

