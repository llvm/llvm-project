// RUN: mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// Test that RegionBranchOpInterface implementations for OpenACC single-region
// ops behave reasonably under LastModifiedAnalysis.


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
// CHECK-NEXT:   - pre
func.func @last_mod_openacc_kernels(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_kernels_before"} : memref<f32>
  acc.kernels {
    acc.terminator
  }
  memref.load %arg0[] {tag = "acc_kernels_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// CHECK-LABEL: test_tag: acc_data_before:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-LABEL: test_tag: acc_data_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
func.func @last_mod_openacc_data(%arg0: memref<f32>, %mapped: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_data_before"} : memref<f32>
  acc.data {
    acc.terminator
  } defaultAttr(none)
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
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "acc_host_before"} : memref<f32>
  %devptr = acc.use_device varPtr(%mapped : memref<f32>) varType(tensor<f32>) -> memref<f32>
  acc.host_data dataOperands(%devptr : memref<f32>) {
    acc.terminator
  }
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
// CHECK-NEXT:   - loop_region
// these bounds run the body at least once, so the store in the loop is the
// only possible last writer
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
  } auto_
  memref.load %arg0[] {tag = "acc_loop_after"} : memref<f32>
  memref.store %zero, %arg0[] {tag_name = "post_loop"} : memref<f32>
  memref.load %arg0[] {tag = "acc_loop_post"} : memref<f32>
  return {tag = "acc_loop_return"} %arg0 : memref<f32>
}

// -----

// structured acc.loop with an unknown upper bound: the body may or may not
// run, so the edge that branches past it is kept.
//
// CHECK-LABEL: test_tag: acc_loop_dynamic_after:
// CHECK:  operand #0
// CHECK-DAG:   - pre
// CHECK-DAG:   - loop_region
func.func @last_mod_openacc_loop_dynamic(%arg0: memref<f32>, %n: i32) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%n : i32)
      step (%c1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_dynamic_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop with matching bounds and an exclusive upper bound: the
// entry test already fails at the lower bound, so the body never runs and the
// store before the loop is the last writer.
//
// CHECK-LABEL: test_tag: acc_loop_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_empty(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c10_i32 : i32) to (%c10_i32 : i32)
      step (%c1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_empty_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop counting down with an inclusive upper bound: `lb` is
// above `ub`, which an ascending-only comparison would misread as an empty
// iteration space. This loop runs 10 times, so the body is guaranteed to run
// and the store inside it is the only possible last writer.
//
// CHECK-LABEL: test_tag: acc_loop_descending_after:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
func.func @last_mod_openacc_loop_descending(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c10_i32 : i32) to (%c1_i32 : i32)
      step (%cm1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_ inclusiveUpperbound(array<i1: true>)
  memref.load %arg0[] {tag = "acc_loop_descending_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop stepping down away from an inclusive upper bound above
// it: the body never runs, so the store before the loop is the last writer.
// Comparing the bounds as if the step were ascending would instead prove the
// body always runs, which is the opposite conclusion.
//
// CHECK-LABEL: test_tag: acc_loop_descending_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_descending_empty(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32)
      step (%cm1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_ inclusiveUpperbound(array<i1: true>)
  memref.load %arg0[] {tag = "acc_loop_descending_empty_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop whose bounds match and whose upper bound is inclusive:
// this runs exactly once. Same bounds as @last_mod_openacc_loop_empty, which
// runs zero times because its upper bound is exclusive, so ignoring
// `inclusiveUpperbound` here would reach the opposite conclusion.
//
// CHECK-LABEL: test_tag: acc_loop_inclusive_single_after:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
func.func @last_mod_openacc_loop_inclusive_single(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c10_i32 : i32) to (%c10_i32 : i32)
      step (%c1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_ inclusiveUpperbound(array<i1: true>)
  memref.load %arg0[] {tag = "acc_loop_inclusive_single_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop ascending with an inclusive upper bound below its lower
// bound: the body never runs.
//
// CHECK-LABEL: test_tag: acc_loop_inclusive_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_inclusive_empty(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c11_i32 = arith.constant 11 : i32
  acc.loop control(%iv : i32) = (%c11_i32 : i32) to (%c10_i32 : i32)
      step (%c1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_ inclusiveUpperbound(array<i1: true>)
  memref.load %arg0[] {tag = "acc_loop_inclusive_empty_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop counting down with an exclusive upper bound: the body
// runs 9 times. Together with @last_mod_openacc_loop_descending this covers
// both upper-bound kinds for a descending step.
//
// CHECK-LABEL: test_tag: acc_loop_descending_exclusive_after:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
func.func @last_mod_openacc_loop_descending_exclusive(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c10_i32 : i32) to (%c1_i32 : i32)
      step (%cm1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_descending_exclusive_after"}
      : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop counting down with matching bounds and an exclusive
// upper bound: the body never runs. The inclusive variant of these same bounds
// would run once, so this is the descending mirror of
// @last_mod_openacc_loop_inclusive_single.
//
// CHECK-LABEL: test_tag: acc_loop_descending_exclusive_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_descending_exclusive_empty(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %cm1_i32 = arith.constant -1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c10_i32 : i32) to (%c10_i32 : i32)
      step (%cm1_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_descending_exclusive_empty_after"}
      : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop whose step overshoots the upper bound: the entry test
// still holds at the lower bound, so the body runs once and no trip count is
// needed to see it.
//
// CHECK-LABEL: test_tag: acc_loop_big_step_after:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
func.func @last_mod_openacc_loop_big_step(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %c100_i32 = arith.constant 100 : i32
  acc.loop control(%iv : i32) = (%c0_i32 : i32) to (%c5_i32 : i32)
      step (%c100_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_big_step_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// structured acc.loop with a zero step: it either never advances or never
// starts, so neither edge can be ruled out.
//
// CHECK-LABEL: test_tag: acc_loop_zero_step_after:
// CHECK:  operand #0
// CHECK-DAG:   - pre
// CHECK-DAG:   - loop_region
func.func @last_mod_openacc_loop_zero_step(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32)
      step (%c0_i32 : i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_zero_step_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// container-like acc.loop: it carries no bounds of its own, so its iteration
// space cannot be inspected here even though the contained scf.for is proven
// to run. Both edges out of the acc.loop are kept.
//
// CHECK-LABEL: test_tag: acc_loop_container_after:
// CHECK:  operand #0
// CHECK-DAG:   - pre
// CHECK-DAG:   - loop_region
func.func @last_mod_openacc_loop_container(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  acc.loop {
    scf.for %i = %c0 to %c10 step %c1 {
      memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    }
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_container_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// collapsed acc.loop whose every dimension runs: the body is entered.
//
// CHECK-LABEL: test_tag: acc_loop_collapsed_after:
// CHECK:  operand #0
// CHECK-NEXT:   - loop_region
func.func @last_mod_openacc_loop_collapsed(%arg0: memref<f32>) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%i : i32, %j : i32) = (%c1_i32, %c1_i32 : i32, i32)
      to (%c10_i32, %c10_i32 : i32, i32) step (%c1_i32, %c1_i32 : i32, i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_collapsed_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// collapsed acc.loop with one empty dimension: the dimensions are iterated as
// a nest, so the whole iteration space is empty even though the first
// dimension would run on its own.
//
// CHECK-LABEL: test_tag: acc_loop_collapsed_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_collapsed_empty(%arg0: memref<f32>)
    -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%i : i32, %j : i32) = (%c1_i32, %c10_i32 : i32, i32)
      to (%c10_i32, %c10_i32 : i32, i32) step (%c1_i32, %c1_i32 : i32, i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_collapsed_empty_after"} : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// collapsed acc.loop pairing an unknown dimension with an empty one: an empty
// dimension zeroes the nest whatever the others do, so the unknown dimension
// does not stop the body from being proven unreachable.
//
// CHECK-LABEL: test_tag: acc_loop_collapsed_unknown_empty_after:
// CHECK:  operand #0
// CHECK-NEXT:   - pre
// CHECK-NOT:   - loop_region
func.func @last_mod_openacc_loop_collapsed_unknown_empty(%arg0: memref<f32>,
    %n: i32) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%i : i32, %j : i32) = (%c1_i32, %c10_i32 : i32, i32)
      to (%n, %c10_i32 : i32, i32) step (%c1_i32, %c1_i32 : i32, i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_collapsed_unknown_empty_after"}
      : memref<f32>
  return %arg0 : memref<f32>
}

// -----

// collapsed acc.loop pairing an unknown dimension with one that runs: the body
// runs only if every dimension runs, so the unknown dimension keeps both edges.
//
// CHECK-LABEL: test_tag: acc_loop_collapsed_unknown_after:
// CHECK:  operand #0
// CHECK-DAG:   - pre
// CHECK-DAG:   - loop_region
func.func @last_mod_openacc_loop_collapsed_unknown(%arg0: memref<f32>,
    %n: i32) -> memref<f32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  memref.store %zero, %arg0[] {tag_name = "pre"} : memref<f32>
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  acc.loop control(%i : i32, %j : i32) = (%c1_i32, %c1_i32 : i32, i32)
      to (%n, %c10_i32 : i32, i32) step (%c1_i32, %c1_i32 : i32, i32) {
    memref.store %one, %arg0[] {tag_name = "loop_region"} : memref<f32>
    acc.yield
  } auto_
  memref.load %arg0[] {tag = "acc_loop_collapsed_unknown_after"} : memref<f32>
  return %arg0 : memref<f32>
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
    memref.store %one, %arg0[] {tag_name = "loop_unstructured_early"} : memref<f32>
    acc.yield

  ^normal_exit:
    memref.store %one, %arg0[] {tag_name = "loop_unstructured_normal"} : memref<f32>
    acc.yield
  } auto_ unstructured
  memref.load %arg0[] {tag = "acc_loop_unstructured_after"} : memref<f32>
  return %arg0 : memref<f32>
}

