// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A vector (thread_x) worksharing loop nested in a SEQUENTIAL loop nested in a
// worker (thread_y) loop reuses worker-private memory across the sequential
// loop's iterations. The vector lanes of each worker row must reconverge before
// the next iteration overwrites the shared slot (WAR). Because worker rows can
// have divergent trip counts, the reconvergence barrier must be PER-ROW
// (gpu.barrier scope<subgroup>), NOT a workgroup-wide barrier (which could deadlock).
//
// On a worker-level parent (no gang ancestor between the seq loop and its
// parent) there is no block-level reconvergence path, so without the worker
// per-row barrier no barrier is emitted here at all.

// CHECK-LABEL: func.func @worker_seq_vector_reuse
// CHECK:       gpu.launch
// Enclosing sequential loop, then the vector loop, then a PER-ROW barrier
// before the next sequential iteration. The barrier must be subgroup-scoped.
// CHECK:         scf.parallel
// CHECK:           scf.parallel
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           {{.*}}par_dims{{.*}}sequential
// CHECK:           gpu.barrier scope <subgroup>
// CHECK-NOT:       gpu.barrier{{[[:space:]]}}
func.func @worker_seq_vector_reuse() {
  %c256_pw = arith.constant 256 : index
  %c4_pw = arith.constant 4 : index
  %c32_pw = arith.constant 32 : index
  %par_bx = acc.par_width %c256_pw {par_dim = #acc.par_dim<block_x>}
  %par_ty = acc.par_width %c4_pw {par_dim = #acc.par_dim<thread_y>}
  %par_tx = acc.par_width %c32_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv0 = acc.privatize : () -> !acc.private_type<memref<32xf32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx) ins(%arg10 = %priv0) : (!acc.private_type<memref<32xf32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 1.0 : f32
      scf.parallel (%gang_iv) = (%c0) to (%grid) step (%c1) {
        scf.parallel (%worker_iv) = (%c0) to (%worker) step (%c1) {
          %pl0 = acc.private_local %arg10 : (!acc.private_type<memref<32xf32>>) -> memref<32xf32>
          // Enclosing sequential loop: the slot is reused across its iterations.
          scf.parallel (%k) = (%c0) to (%c8) step (%c1) {
            // Vector (thread_x) loop with cross-lane reuse of worker-private memory.
            scf.parallel (%vec_iv0) = (%c0) to (%block) step (%c1) {
              scf.parallel (%seq0) = (%vec_iv0) to (%c32) step (%block) {
                %v = memref.load %pl0[%seq0] : memref<32xf32>
                memref.store %v, %pl0[%seq0] : memref<32xf32>
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
