// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=0}))' -split-input-file

// XFAIL: *

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {}
  return
}
