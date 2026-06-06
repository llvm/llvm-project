// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=0,0}))' -split-input-file -verify-diagnostics

// The expected error is, "tile size cannot be 0" at an unknown location. (It's 
// location is unknown because the it's caused by an invalid command line 
// argument.)
// XFAIL: *

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) {}
  return
}
