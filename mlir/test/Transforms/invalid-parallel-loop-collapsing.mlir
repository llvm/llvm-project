// First test various sets of invalid arguments
// RUN: not mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing))' 2>&1 | FileCheck %s --check-prefix=CL0
// CL0: No collapsed-indices were specified. This pass is only for testing and does not automatically collapse all parallel loops or similar

// RUN: not mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-1=1}))' 2>&1 | FileCheck %s --check-prefix=CL1
// CL1: collapsed-indices-1 specified but not collapsed-indices-0

// RUN: not mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=1  collapsed-indices-2=2}))' 2>&1 | FileCheck %s --check-prefix=CL2
// CL2: collapsed-indices-2 specified but not collapsed-indices-1

// RUN: not mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=1  collapsed-indices-1=2}))' 2>&1 | FileCheck %s --check-prefix=NON-ZERO
// NON-ZERO: collapsed-indices arguments must include all values [0,N).

// RUN: not mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0  collapsed-indices-1=2}))' 2>&1 | FileCheck %s --check-prefix=NON-CONTIGUOUS
// NON-CONTIGUOUS: collapsed-indices arguments must include all values [0,N).


// Then test for invalid combinations of argument+input-ir
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}))' -verify-diagnostics
func.func @too_few_iters(%arg0: index, %arg1: index, %arg2: index) {
  // expected-error @+1 {{op has 1 iter args while this limited functionality testing pass was configured only for loops with exactly 2 iter args.}}
  scf.parallel (%arg3) = (%arg0) to (%arg1) step (%arg2) {
    scf.yield
  }
  return
}

func.func @too_many_iters(%arg0: index, %arg1: index, %arg2: index) {
  // expected-error @+1 {{op has 3 iter args while this limited functionality testing pass was configured only for loops with exactly 2 iter args.}}
  scf.parallel (%arg3, %arg4, %arg5) = (%arg0, %arg0, %arg0) to (%arg1, %arg1, %arg1) step (%arg2, %arg2, %arg2) {
    scf.yield
  }
  return
}
