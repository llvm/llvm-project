// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
// Ensure that the -mlir-to-llmvir pass doesn't crash.

// CHECK: define void @_QQmain()

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false} {
  llvm.func @_QQmain()  {
    omp.target {
      %0 = llvm.mlir.constant(1000 : i32) : i32
      %1 = llvm.mlir.constant(1 : i32) : i32
      omp.teams {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg0) : i32 = (%1) to (%0) inclusive step (%1) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
