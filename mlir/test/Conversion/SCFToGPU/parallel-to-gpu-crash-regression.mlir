// RUN: mlir-opt %s --convert-parallel-loops-to-gpu | FileCheck %s

// Goal: exercise the per-dim index computation
//        newIndex = hardware_id * step + lowerBound
// and ensure we see a gpu.launch and an affine.apply (no crash).

module {
  func.func @two_dim_parallel_mapped() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    // Single 2‑D scf.parallel. Each dimension is mapped to a GPU dim.
    // We *use* both IVs so the conversion must build indices.
    scf.parallel (%bx, %tx) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
      %u = arith.addi %bx, %c0 : index
      %v = arith.addi %tx, %c0 : index
      // No explicit terminator: the parser inserts an empty scf.reduce.
    } {
      mapping = [
        #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
        #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>
      ]
    }
    return
  }
}

// CHECK-LABEL: func.func @two_dim_parallel_mapped
// CHECK:       gpu.launch
// CHECK:       affine.apply
