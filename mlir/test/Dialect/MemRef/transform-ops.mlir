// RUN: mlir-opt %s -test-transform-dialect-interpreter -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  scf.for %i0 = %c0 to %c16 step %c4 {
    // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
    // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    // CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4xf32, #[[$MAP1]]> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64}
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !pdl.operation
}
