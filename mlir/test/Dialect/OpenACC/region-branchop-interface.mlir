// RUN: mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// Test that RegionBranchOpInterface implementations for OpenACC single-region
// ops behave reasonably under LastModifiedAnalysis.
//
// For acc.parallel / acc.serial, which only have value-based memory effects
// coming from their bodies, the analysis can track the last writer precisely.
// For other OpenACC region ops (e.g. acc.kernels, acc.data, acc.host_data),
// which currently only report resource-only memory effects at the op level,
// LastModifiedAnalysis cannot attribute a precise last writer and falls back
// to "<unknown>" after the op for values not directly touched in the region.


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

// -----

// CHECK-LABEL: test_tag: acc_kernels_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_kernels_after:
// CHECK:  operand #0
// CHECK-NEXT:   - <unknown>
func.func @last_mod_openacc_kernels(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32

  // Single store before the region.
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_kernels_before"} : memref<f32>

  // The acc.kernels region does not touch %arg0.
  acc.kernels {
    "test.openacc_dummy_op"() : () -> ()
    acc.terminator
  }

  // After acc.kernels, LastModifiedAnalysis cannot prove that "pre" still
  // dominates all paths and therefore reports the last modifier conservatively
  // as "<unknown>" for this load.
  memref.load %arg0[] {tag = "acc_kernels_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// CHECK-LABEL: test_tag: acc_data_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_data_after:
// CHECK:  operand #0
// CHECK-NEXT:   - <unknown>
func.func @last_mod_openacc_data(%arg0: memref<f32>, %mapped: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32

  // Single store to %arg0 before the region.
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_data_before"} : memref<f32>

  // Map an unrelated buffer into device memory and run an acc.data region that
  // does not touch %arg0.
  %create = acc.create varPtr(%mapped : memref<f32>) varType(tensor<f32>) -> memref<f32>
  acc.data dataOperands(%create : memref<f32>) {
    "test.openacc_dummy_op"() : () -> ()
    acc.terminator
  }

  // After acc.data, LastModifiedAnalysis cannot prove that "pre" still
  // dominates all paths for %arg0 and reports "<unknown>" for this load.
  memref.load %arg0[] {tag = "acc_data_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// CHECK-LABEL: test_tag: acc_host_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_host_after:
// CHECK:  operand #0
// CHECK-NEXT:   - <unknown>
func.func @last_mod_openacc_host_data(%arg0: memref<f32>, %mapped: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32

  // Single store to %arg0 before the region.
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_host_before"} : memref<f32>

  // Map %mapped into device memory and run an acc.host_data region that does
  // not touch %arg0.
  %devptr = acc.use_device varPtr(%mapped : memref<f32>) varType(tensor<f32>) -> memref<f32>
  acc.host_data dataOperands(%devptr : memref<f32>) {
    "test.openacc_dummy_op"() : () -> ()
    acc.terminator
  }

  // After acc.host_data, the analysis cannot prove "pre" still dominates and
  // reports "<unknown>" for the last writer on %arg0.
  memref.load %arg0[] {tag = "acc_host_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop: the RegionBranch is modeled
// as scf.for with a backedge to the parent op.
// CHECK-LABEL: test_tag: acc_loop_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_loop_inside:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
// CHECK-LABEL: test_tag: acc_loop_after:
// CHECK:  operand #0
// CHECK-DAG:   - pre
// CHECK-DAG:   - loop_region
// the last writer is either the pre-loop store or
// the store in the loop depending on the iteration count
// CHECK-LABEL: test_tag: acc_loop_post:
// CHECK:  operand #0
// CHECK-NEXT:   - post_loop
// CHECK-LABEL: test_tag: acc_loop_return:
// CHECK:  operand #0
// CHECK-NEXT:   - post_loop
func.func @last_mod_openacc_loop(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_loop_before"} : memref<f32>
  %one = arith.constant 1.0 : f32
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32)
      step (%c1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    memref.load %arg0[] {tag = "acc_loop_inside"} : memref<f32>
    acc.yield
  } attributes {auto_ = [#acc.device_type<none>],
                inclusiveUpperbound = array<i1: true>}
  memref.load %arg0[] {tag = "acc_loop_after"} : memref<f32>
  memref.store %zero, %arg0[] {tag_name = "post_loop"} : memref<f32>
  memref.load %arg0[] {tag = "acc_loop_post"} : memref<f32>
  return {tag = "acc_loop_return"} %arg0 : memref<f32>
}

// -----

// Unstructured acc.loop: the RegionBranch is modeled with explicit CFG and early
// exits, and the RegionBranch graph only exposes a single entry and single
// exit edge (no region backedge).
//
// CHECK-LABEL: test_tag: acc_loop_unstructured_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_loop_unstructured_after:
// CHECK:  operand #0
// CHECK-DAG:   - loop_unstructured_early
// CHECK-DAG:   - loop_unstructured_normal
// the last writer can be either of the two stores in the loop
func.func @last_mod_openacc_loop_unstructured(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one  = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_loop_unstructured_before"} : memref<f32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c5_i32 = arith.constant 5 : i32
  acc.loop {
  ^entry:
    cf.br ^header(%c0_i32 : i32)

  ^header(%iv: i32):
    %is_early = arith.cmpi eq, %iv, %c1_i32 : i32
    cf.cond_br %is_early, ^early_exit, ^cont

  ^cont:
    // Normal loop increment and exit when iv reaches 5.
    %iv_next = arith.addi %iv, %c1_i32 : i32
    %is_done = arith.cmpi eq, %iv_next, %c5_i32 : i32
    cf.cond_br %is_done, ^normal_exit, ^header(%iv_next : i32)

  ^early_exit:
    // One exit path with its own store.
    memref.store %one, %arg0[] {tag_name = "loop_unstructured_early"} : memref<f32>
    acc.yield

  ^normal_exit:
    // Another exit path with a different store.
    memref.store %one, %arg0[] {tag_name = "loop_unstructured_normal"} : memref<f32>
    acc.yield
  } attributes {auto_ = [#acc.device_type<none>], unstructured}
  memref.load %arg0[] {tag = "acc_loop_unstructured_after"} : memref<f32>
  return %arg0 : memref<f32>
}

