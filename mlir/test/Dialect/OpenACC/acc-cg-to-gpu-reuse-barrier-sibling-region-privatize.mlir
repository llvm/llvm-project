// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// Two sibling acc.compute_regions share one enclosing sequential scf.for loop.
// When lowering the region that stores to gang-private memory,
// needsPreStoreReuseBarrier calls findFirstSequentialLoop, which walks up past
// the compute region to the shared scf.for and then walks its whole body. That
// walk descends into the sibling compute region and reaches a memref.load whose
// acc.private_local is defined there: its getPrivatized() value is a block
// argument of the sibling region, not the region being lowered. getPrivatizeOp
// used to map that block arg through the region being lowered, so
// ComputeRegionOp::getOperand returned a null Value and getDefiningOp() aborted
// (dyn_cast on a non-existent value). Both regions must lower without crashing.

// CHECK-LABEL: func.func @sibling_regions_share_seq_loop
// CHECK:         scf.for
// CHECK:           gpu.launch
// CHECK:             memref.load
// CHECK:           gpu.launch
// CHECK:             gpu.barrier
// CHECK:             gpu.barrier
// CHECK:             memref.load
func.func @sibling_regions_share_seq_loop() {
  %c256_pw = arith.constant 256 : index
  %c1024_pw = arith.constant 1024 : index
  %par_bx = acc.par_width %c256_pw {par_dim = #acc.par_dim<block_x>}
  %par_tx = acc.par_width %c1024_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv0 = acc.privatize : () -> !acc.private_type<memref<i64>>
    %priv1 = acc.privatize : () -> !acc.private_type<memref<i64>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %t = %c0 to %c8 step %c1 {
      // Sibling region: reads its own gang-private local.
      acc.compute_region launch(%gridB = %par_bx, %blockB = %par_tx) ins(%argB = %priv1) : (!acc.private_type<memref<i64>>) {
        %cb0 = arith.constant 0 : index
        %cb1 = arith.constant 1 : index
        scf.parallel (%gang_iv2) = (%cb0) to (%gridB) step (%cb1) {
          %plB = acc.private_local %argB : (!acc.private_type<memref<i64>>) -> memref<i64>
          %vb = memref.load %plB[] : memref<i64>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        acc.yield
      } {origin = "acc.parallel"}
      // Region with a gang-private store inside a block-level predicate region.
      acc.compute_region launch(%gridA = %par_bx, %blockA = %par_tx) ins(%argA = %priv0) : (!acc.private_type<memref<i64>>) {
        %ca0 = arith.constant 0 : index
        %ca1 = arith.constant 1 : index
        scf.parallel (%gang_iv) = (%ca0) to (%gridA) step (%ca1) {
          %plA = acc.private_local %argA : (!acc.private_type<memref<i64>>) -> memref<i64>
          acc.predicate_region {
            %val = arith.index_cast %gang_iv : index to i64
            memref.store %val, %plA[] : memref<i64>
          }
          %v0 = memref.load %plA[] : memref<i64>
          scf.parallel (%vec_iv) = (%ca0) to (%blockA) step (%ca1) {
            memref.store %v0, %plA[] : memref<i64>
            scf.reduce
          } {acc.par_dims = #acc<par_dims[thread_x]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        acc.yield
      } {origin = "acc.parallel"}
    }
  }
  return
}
