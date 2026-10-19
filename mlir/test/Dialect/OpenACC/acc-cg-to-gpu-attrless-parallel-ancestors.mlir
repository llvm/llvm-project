// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu),gpu.module(gpu.func(acc-cg-to-gpu)))" \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s \
// RUN:   --implicit-check-not=acc.compute_region --implicit-check-not=gpu.barrier

// Parallel dimensions must be assigned before lowering a compute region.
// Reject an unmapped parallel loop even without a nested sequential loop that
// would otherwise trigger a barrier lookup.

func.func @attrless_parallel_in_compute_region() {
  acc.compute_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{requires an 'acc.par_dims' attribute}}
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    }
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A mapped enclosing loop does not make an unmapped nested loop valid. Check
// recursively through non-loop regions as well.

func.func @attrless_parallel_nested_in_if() {
  acc.compute_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
      scf.if %true {
        // expected-error@+1 {{requires an 'acc.par_dims' attribute}}
        scf.parallel (%j) = (%c0) to (%c1) step (%c1) {
        }
      }
    } {acc.par_dims = #acc<par_dims[block_x]>}
  } <{origin = "acc.parallel"}>
  return
}

// -----

// The same input requirement applies to compute regions in GPU functions.

module attributes {gpu.container_module} {
  gpu.module @device {
    gpu.func @attrless_parallel_in_gpu_func() {
      acc.compute_region {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // expected-error@+1 {{requires an 'acc.par_dims' attribute}}
        scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
        }
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// Parallel loops outside the compute region do not need GPU mappings. Barrier
// lookup for the plain sequential loop must not inspect these host ancestors.

// CHECK-LABEL: func.func @attrless_host_parallel_ancestors
// CHECK: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK-NEXT: scf.parallel
// CHECK: gpu.launch
// CHECK: scf.for
func.func @attrless_host_parallel_ancestors() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    scf.parallel (%j) = (%c0) to (%c1) step (%c1) {
      scf.parallel (%k) = (%c0) to (%c1) step (%c1) {
        acc.compute_region {
          %c0_inner = arith.constant 0 : index
          %c1_inner = arith.constant 1 : index
          scf.for %l = %c0_inner to %c1_inner step %c1_inner {
          }
        } <{origin = "acc.parallel"}>
      }
    }
  }
  return
}
