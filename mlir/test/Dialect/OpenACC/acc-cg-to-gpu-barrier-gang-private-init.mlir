// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// A gang-redundant init loop that writes a gang-private array, followed by a
// loop that reads it, must be separated by a workgroup barrier.

// CHECK-LABEL: @test_gang_private_init_barrier
// CHECK: gpu.launch
// CHECK: memref.store {{.*}} memref<8xf32>
// CHECK: gpu.barrier
// CHECK: scf.parallel

func.func @test_gang_private_init_barrier(%arg0: memref<100x100xf32>) {
  %c128 = arith.constant 128 : index
  %c10 = arith.constant 10 : index
  %c8 = arith.constant 8 : index
  %c100 = arith.constant 100 : index
  %12 = acc.copyin varPtr(%arg0 : memref<100x100xf32>) -> memref<100x100xf32> {dataClause = #acc<data_clause acc_copy>, name = "gp"}
  %13 = acc.par_width %c10 {par_dim = #acc.par_dim<block_x>}
  %14 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment dataOperands(%12 : memref<100x100xf32>) {
    %15 = acc.privatize [#acc<par_dims[block_x]>] : () -> !acc.private_type<memref<8xf32>>
    acc.compute_region launch(%arg1 = %13, %arg2 = %14) ins(%arg10 = %12, %arg11 = %15) : (memref<100x100xf32>, !acc.private_type<memref<8xf32>>) {
      %c8_b = arith.constant 8 : index
      %c100_b = arith.constant 100 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %19 = acc.private_local %arg11 : (!acc.private_type<memref<8xf32>>) -> memref<8xf32>
      scf.parallel (%tx) = (%c0) to (%c8_b) step (%c1) {
        scf.parallel (%s) = (%c0) to (%c8_b) step (%c1) {
          %v = arith.index_cast %s : index to i32
          %vf = arith.sitofp %v : i32 to f32
          memref.store %vf, %19[%s] : memref<8xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      scf.parallel (%bx2) = (%c0) to (%c100_b) step (%c1) {
        scf.parallel (%tx2) = (%c0) to (%c100_b) step (%c1) {
          scf.parallel (%s2) = (%c0) to (%c8_b) step (%c1) {
            %r = memref.load %19[%s2] : memref<8xf32>
            acc.predicate_region {
              memref.store %r, %arg10[%bx2, %tx2] : memref<100x100xf32>
            }
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// A thread-private array must not introduce a workgroup barrier between init and read.

// CHECK-LABEL: @test_thread_private_init_no_barrier
// CHECK: memref.store {{.*}} memref<8xf32>
// CHECK-NOT: gpu.barrier
// CHECK: scf.parallel

func.func @test_thread_private_init_no_barrier(%arg0: memref<100x100xf32>) {
  %c128 = arith.constant 128 : index
  %c10 = arith.constant 10 : index
  %c8 = arith.constant 8 : index
  %c100 = arith.constant 100 : index
  %12 = acc.copyin varPtr(%arg0 : memref<100x100xf32>) -> memref<100x100xf32> {dataClause = #acc<data_clause acc_copy>, name = "gp"}
  %13 = acc.par_width %c10 {par_dim = #acc.par_dim<block_x>}
  %14 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment dataOperands(%12 : memref<100x100xf32>) {
    %15 = acc.privatize [#acc<par_dims[block_x, thread_x]>] : () -> !acc.private_type<memref<8xf32>>
    acc.compute_region launch(%arg1 = %13, %arg2 = %14) ins(%arg10 = %12, %arg11 = %15) : (memref<100x100xf32>, !acc.private_type<memref<8xf32>>) {
      %c8_b = arith.constant 8 : index
      %c100_b = arith.constant 100 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %19 = acc.private_local %arg11 : (!acc.private_type<memref<8xf32>>) -> memref<8xf32>
      scf.parallel (%tx) = (%c0) to (%c8_b) step (%c1) {
        scf.parallel (%s) = (%c0) to (%c8_b) step (%c1) {
          %v = arith.index_cast %s : index to i32
          %vf = arith.sitofp %v : i32 to f32
          memref.store %vf, %19[%s] : memref<8xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      scf.parallel (%bx2) = (%c0) to (%c100_b) step (%c1) {
        scf.parallel (%tx2) = (%c0) to (%c100_b) step (%c1) {
          scf.parallel (%s2) = (%c0) to (%c8_b) step (%c1) {
            %r = memref.load %19[%s2] : memref<8xf32>
            acc.predicate_region {
              memref.store %r, %arg10[%bx2, %tx2] : memref<100x100xf32>
            }
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
