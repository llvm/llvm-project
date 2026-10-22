// RUN: %python %S/Inputs/gemm_data.py %t 128 128 64
// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --cpu agx-g14c --grid 512 --wg 128 %t.mlir f32:@%t/a.txt \
// RUN:   f32:@%t/b.txt f32*16384 index=128 index=128 index=64 \
// RUN:   | %python %S/Inputs/gemm_data.py --check %t/c.txt | FileCheck %s
// RUN: %run_agx --cpu agx-g13g --grid 512 --wg 128 %t.mlir f32:@%t/a.txt \
// RUN:   f32:@%t/b.txt f32*16384 index=128 index=128 index=64 \
// RUN:   | %python %S/Inputs/gemm_data.py --check %t/c.txt | FileCheck %s

// CHECK: gemm: 0 of 16384 elements wrong

// C = A * B for row-major f32 matrices, M and N multiples of 64 and K a
// multiple of 8. The same source serves every AGX generation: the chip of the
// module's #agxir.target decides how agx.mma_tile and agx.block_barrier run.
//
// Each threadgroup is 2 x 2 SIMD-groups (128 threads) computing a 64 x 64
// tile; each SIMD-group a 32 x 32 tile, accumulated in registers along K with
// the operands read straight from device memory (Apple's guidance for its
// GPUs: no threadgroup staging, enough occupancy instead of software
// pipelining). Threadgroups walk the output in groups of 8 tile rows so that
// concurrently running ones share rows of A and columns of B in the caches,
// and the SIMD-groups of a threadgroup meet at a barrier every 128 steps of K
// to keep their working set together.

gpu.module @kernels {
  gpu.func @agx_gemm_f32(%a0: memref<1xf32, 1>, %b0: memref<1xf32, 1>,
                         %c0: memref<1xf32, 1>, %M: index, %N: index,
                         %K: index) kernel {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i8 = arith.constant 8 : index
    %i32 = arith.constant 32 : index
    %i64 = arith.constant 64 : index
    %i128 = arith.constant 128 : index
    %A = memref.reinterpret_cast %a0 to offset: [0], sizes: [%M, %K], strides: [%K, 1]
        : memref<1xf32, 1> to memref<?x?xf32, strided<[?, 1]>, 1>
    %B = memref.reinterpret_cast %b0 to offset: [0], sizes: [%K, %N], strides: [%N, 1]
        : memref<1xf32, 1> to memref<?x?xf32, strided<[?, 1]>, 1>
    %C = memref.reinterpret_cast %c0 to offset: [0], sizes: [%M, %N], strides: [%N, 1]
        : memref<1xf32, 1> to memref<?x?xf32, strided<[?, 1]>, 1>

    // Grouped tile order: groups of 8 tile rows, column-major inside a group.
    %tiles_m = arith.divui %M, %i64 : index
    %tiles_n = arith.divui %N, %i64 : index
    %id = gpu.block_id x
    %per_group = arith.muli %tiles_n, %i8 : index
    %group = arith.divui %id, %per_group : index
    %first_m = arith.muli %group, %i8 : index
    %left_m = arith.subi %tiles_m, %first_m : index
    %group_m = arith.minui %left_m, %i8 : index
    %in_group = arith.remui %id, %per_group : index
    %rem_m = arith.remui %in_group, %group_m : index
    %tm = arith.addi %first_m, %rem_m : index
    %tn = arith.divui %in_group, %group_m : index

    // This SIMD-group's 32 x 32 tile.
    %sg = gpu.subgroup_id : index
    %sr = arith.divui %sg, %i2 : index
    %sc = arith.remui %sg, %i2 : index
    %tm64 = arith.muli %tm, %i64 : index
    %tn64 = arith.muli %tn, %i64 : index
    %sr32 = arith.muli %sr, %i32 : index
    %sc32 = arith.muli %sc, %i32 : index
    %row = arith.addi %tm64, %sr32 : index
    %col = arith.addi %tn64, %sc32 : index

    %zero = arith.constant dense<0.0> : vector<32xf32>
    %acc = scf.for %kk = %i0 to %K step %i128 iter_args(%acc_o = %zero) -> vector<32xf32> {
      agx.block_barrier
      %kk_end0 = arith.addi %kk, %i128 : index
      %kk_end = arith.minui %kk_end0, %K : index
      %acc_i = scf.for %k = %kk to %kk_end step %i8 iter_args(%acc_k = %acc_o) -> vector<32xf32> {
        %next = agx.mma_tile %A[%row, %k] * %B[%k, %col] + %acc_k tile(32, 32, 8)
            : memref<?x?xf32, strided<[?, 1]>, 1>, memref<?x?xf32, strided<[?, 1]>, 1>, vector<32xf32>
        scf.yield %next : vector<32xf32>
      }
      scf.yield %acc_i : vector<32xf32>
    }

    // The 16 fragments of the tile, row-major, two elements per thread.
    %i4 = arith.constant 4 : index
    %i16 = arith.constant 16 : index
    scf.for %f = %i0 to %i16 step %i1 {
      %f_r = arith.divui %f, %i4 : index
      %f_c = arith.remui %f, %i4 : index
      %r_o = arith.muli %f_r, %i8 : index
      %c_o = arith.muli %f_c, %i8 : index
      %r = arith.addi %row, %r_o : index
      %c = arith.addi %col, %c_o : index
      %e0 = arith.muli %f, %i2 : index
      %e1 = arith.addi %e0, %i1 : index
      %x0 = vector.extract %acc[%e0] : f32 from vector<32xf32>
      %x1 = vector.extract %acc[%e1] : f32 from vector<32xf32>
      %frag = vector.from_elements %x0, %x1 : vector<2xf32>
      agx.matrix_store %frag, %C[%r, %c] : vector<2xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    }
    gpu.return
  }
}
