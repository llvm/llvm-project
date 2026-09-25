// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" \
// RUN:   --split-input-file | FileCheck %s \
// RUN:   --implicit-check-not=gpu.barrier --implicit-check-not=acc.compute_region

// Unmapped parallel loops are preserved by the generic region lowering. The
// barrier lookup after the sequential loop must tolerate an ancestor without
// acc.par_dims.

// CHECK-LABEL: func.func @attrless_parallel_ancestors
// CHECK: gpu.launch
// CHECK: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK-NEXT: scf.for
func.func @attrless_parallel_ancestors() {
  acc.compute_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
      scf.parallel (%j) = (%c0) to (%c1) step (%c1) {
        scf.parallel (%k) = (%c0) to (%c1) step (%c1) {
          scf.for %l = %c0 to %c1 step %c1 {
          }
        }
      }
    }
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A sequential parallel parent takes a different barrier lookup path when its
// child loop contains thread-level work. The potential block-level ancestor
// can still be an unmapped parallel loop with no acc.par_dims.

// CHECK-LABEL: func.func @sequential_parent_with_attrless_ancestor
// CHECK: gpu.launch
// CHECK: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK-NEXT: scf.for
func.func @sequential_parent_with_attrless_ancestor() {
  acc.compute_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
      scf.parallel (%j) = (%c0) to (%c1) step (%c1) {
        scf.parallel (%k) = (%c0) to (%c1) step (%c1) {
          scf.for %l = %c0 to %c1 step %c1 {
            scf.parallel (%tx) = (%c0) to (%c1) step (%c1) {
            } {acc.par_dims = #acc<par_dims[thread_x]>}
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
      }
    }
  } <{origin = "acc.parallel"}>
  return
}
