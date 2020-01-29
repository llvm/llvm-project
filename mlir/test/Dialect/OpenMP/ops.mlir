// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s

func @omp_barrier() -> () {
  // CHECK: omp.barrier
  omp.barrier
  return
}
