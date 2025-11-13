// RUN: mlir-opt %s --convert-parallel-loops-to-gpu | FileCheck %s

module {
  func.func @one_dim_parallel_mapped() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    // 1‑D loop mapped to thread_x; use the IV to force index computation.
    scf.parallel (%t) = (%c0) to (%c64) step (%c1) {
      %w = arith.addi %t, %c0 : index
      // Implicit empty scf.reduce terminator.
    } {
      mapping = [
        #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>
      ]
    }
    return
  }
}

// CHECK-LABEL: func.func @one_dim_parallel_mapped
// CHECK:       gpu.launch
// CHECK:       affine.apply
