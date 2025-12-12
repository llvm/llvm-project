// RUN: mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// Test that RegionBranchOpInterface implementations for OpenACC single-region
// ops (here: acc.parallel and acc.serial) allow LastModifiedAnalysis to
// propagate modifications performed inside the region to uses after the op.


// CHECK-LABEL: test_tag: acc_parallel_after:
// CHECK:  operand #0
// CHECK-NEXT:   - parallel_region
// CHECK-LABEL: test_tag: acc_parallel_return:
// CHECK:  operand #0
// CHECK-NEXT:   - parallel_region
func.func @last_mod_openacc_parallel(%arg0: memref<f32>) -> memref<f32> {
  %one = arith.constant 1.0 : f32

  // The only store to %arg0 happens inside the acc.parallel region.
  acc.parallel {
    memref.store %one, %arg0[] {tag_name = "parallel_region"} : memref<f32>
    acc.yield
  }

  // With RegionBranchOpInterface wired up, the last modification at this load
  // is the store inside the acc.parallel region.
  memref.load %arg0[] {tag = "acc_parallel_after"} : memref<f32>

  // And the same store should be seen at the function return.
  return {tag = "acc_parallel_return"} %arg0 : memref<f32>
}

// -----

// CHECK-LABEL: test_tag: acc_serial_after:
// CHECK:  operand #0
// CHECK-NEXT:   - serial_region
// CHECK-LABEL: test_tag: acc_serial_return:
// CHECK:  operand #0
// CHECK-NEXT:   - serial_region
func.func @last_mod_openacc_serial(%arg0: memref<f32>) -> memref<f32> {
  %one = arith.constant 1.0 : f32

  // The only store to %arg0 happens inside the acc.serial region.
  acc.serial {
    memref.store %one, %arg0[] {tag_name = "serial_region"} : memref<f32>
    acc.yield
  }

  memref.load %arg0[] {tag = "acc_serial_after"} : memref<f32>

  return {tag = "acc_serial_return"} %arg0 : memref<f32>
}
