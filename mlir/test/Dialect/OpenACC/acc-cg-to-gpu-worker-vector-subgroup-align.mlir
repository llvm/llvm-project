// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A worker per-row reconvergence barrier requires blockDim.x to be
// subgroup-aligned, otherwise a warp spans two worker rows and the warp-scoped
// barrier deadlocks under row-divergent control flow.
//
// Here thread_x = 16 (< subgroupSize 32) and thread_y = 4, and the vector loop
// is in a sequential loop with a worker-row-dependent trip count. The fast-path
// per-row barrier (gpu.barrier scope<subgroup>) must trigger the blockDim.x
// alignment: blockDim.x is padded to 32 and blockDim.y reduced to 2, so each
// warp holds exactly one worker row.

// CHECK-LABEL: func.func @worker_divergent_subgroup_barrier
// CHECK:       gpu.launch
// CHECK-SAME:  threads({{.*}}) in (%{{[a-z0-9_]+}} = %c32, %{{[a-z0-9_]+}} = %c2,
// CHECK:       gpu.barrier scope <subgroup>
func.func @worker_divergent_subgroup_barrier() {
  %c256_pw = arith.constant 256 : index
  %c4_pw = arith.constant 4 : index
  %c16_pw = arith.constant 16 : index
  %par_bx = acc.par_width %c256_pw {par_dim = #acc.par_dim<block_x>}
  %par_ty = acc.par_width %c4_pw {par_dim = #acc.par_dim<thread_y>}
  %par_tx = acc.par_width %c16_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv0 = acc.privatize : () -> !acc.private_type<memref<64xf32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx) ins(%arg10 = %priv0) : (!acc.private_type<memref<64xf32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 1.0 : f32
      scf.parallel (%gang_iv) = (%c0) to (%grid) step (%c1) {
        scf.parallel (%worker_iv) = (%c0) to (%worker) step (%c1) {
          %pl0 = acc.private_local %arg10 : (!acc.private_type<memref<64xf32>>) -> memref<64xf32>
          // Worker-row-dependent trip count => rows sharing a warp diverge.
          %ub = arith.addi %worker_iv, %c1 : index
          scf.parallel (%k) = (%c0) to (%ub) step (%c1) {
            scf.parallel (%vec_iv0) = (%c0) to (%block) step (%c1) {
              scf.parallel (%seq0) = (%vec_iv0) to (%c64) step (%block) {
                %v = memref.load %pl0[%seq0] : memref<64xf32>
                memref.store %v, %pl0[%seq0] : memref<64xf32>
                scf.reduce
              } {acc.par_dims = #acc<par_dims[sequential]>}
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
