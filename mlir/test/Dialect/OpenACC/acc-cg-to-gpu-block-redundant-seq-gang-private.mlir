// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A sequential gpu_block_redundant outer loop wrapping a partitioned gang loop
// (block_x + sequential remainder) with gang-private storage. Launch dims must
// not be injected from that sequential redundant wrapper into privatization:
// the private must stay gang-scoped and lower to shared memory.

// CHECK-LABEL: func.func @block_redundant_seq_gang_private
// Gang-scoped private: no launch-dim widening on privatize (bare form), shared storage.
// CHECK:         acc.privatize : () -> !acc.private_type<memref<32xi32>>
// CHECK:         gpu.launch
// CHECK:           acc.gpu_shared_memory

func.func @block_redundant_seq_gang_private(%arg0: memref<32xi32>) {
  %c32_pw = arith.constant 32 : index
  %c1_pw = arith.constant 1 : index
  %par_bx = acc.par_width %c32_pw {par_dim = #acc.par_dim<block_x>}
  %par_ty = acc.par_width %c1_pw {par_dim = #acc.par_dim<thread_y>}
  %par_tx = acc.par_width %c32_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<32xi32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arr = %arg0, %arg10 = %priv)
        : (memref<32xi32>, !acc.private_type<memref<32xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      // Outer gang-redundant loop: sequential over all blocks.
      scf.parallel (%ni) = (%c0) to (%c4) step (%c1) {
        // Partitioned gang: block_x wrapping a sequential grid-stride remainder.
        scf.parallel (%g) = (%c0) to (%grid) step (%c1) {
          scf.parallel (%grem) = (%g) to (%c32) step (%grid) {
            %work = acc.private_local %arg10
                : (!acc.private_type<memref<32xi32>>) -> memref<32xi32>
            scf.parallel (%v) = (%c0) to (%block) step (%c1) {
              %ld = memref.load %arr[%v] : memref<32xi32>
              memref.store %ld, %work[%v] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.parallel (%v2) = (%c0) to (%block) step (%c1) {
              %val = memref.load %work[%v2] : memref<32xi32>
              memref.store %val, %arr[%v2] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>,
         acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
