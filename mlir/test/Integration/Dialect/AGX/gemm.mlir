// RUN: %python %S/Inputs/gemm_data.py %t 128 128 64
// RUN: mlir-opt %s -convert-gpu-to-air='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --cpu agx-g14c --grid 512 --wg 128 %t.mlir f32:@%t/a.txt \
// RUN:   f32:@%t/b.txt f32*16384 index=128 index=128 index=64 \
// RUN:   | %python %S/Inputs/gemm_data.py --check %t/c.txt | FileCheck %s
// RUN: %run_agx --cpu agx-g13g --grid 512 --wg 128 %t.mlir f32:@%t/a.txt \
// RUN:   f32:@%t/b.txt f32*16384 index=128 index=128 index=64 \
// RUN:   | %python %S/Inputs/gemm_data.py --check %t/c.txt | FileCheck %s

// CHECK: gemm: 0 of 16384 elements wrong

// C = A * B for row-major f32 matrices, M and N multiples of 64 and K a
// multiple of 8. The same source serves every AGX generation: air 8x8
// SIMD-group matrices, which the AGX backend compiles for the chip.
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

    // 4 x 4 8x8 matrices.
    %i8x = arith.constant 8 : index
    %row1 = arith.addi %row, %i8x : index
    %col1 = arith.addi %col, %i8x : index
    %i16x = arith.constant 16 : index
    %row2 = arith.addi %row, %i16x : index
    %col2 = arith.addi %col, %i16x : index
    %i24x = arith.constant 24 : index
    %row3 = arith.addi %row, %i24x : index
    %col3 = arith.addi %col, %i24x : index
    %fz = arith.constant 0.0 : f32
    %zero = air.simdgroup_matrix_fill %fz : vector<64xf32>
    %acc:16 = scf.for %kk = %i0 to %K step %i128
        iter_args(%o00 = %zero, %o01 = %zero, %o02 = %zero, %o03 = %zero, %o10 = %zero, %o11 = %zero, %o12 = %zero, %o13 = %zero, %o20 = %zero, %o21 = %zero, %o22 = %zero, %o23 = %zero, %o30 = %zero, %o31 = %zero, %o32 = %zero, %o33 = %zero) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
      gpu.barrier memfence []
      %kk_end0 = arith.addi %kk, %i128 : index
      %kk_end = arith.minui %kk_end0, %K : index
      %in:16 = scf.for %k = %kk to %kk_end step %i8
          iter_args(%a00 = %o00, %a01 = %o01, %a02 = %o02, %a03 = %o03, %a10 = %o10, %a11 = %o11, %a12 = %o12, %a13 = %o13, %a20 = %o20, %a21 = %o21, %a22 = %o22, %a23 = %o23, %a30 = %o30, %a31 = %o31, %a32 = %o32, %a33 = %o33) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
        %ma0 = air.simdgroup_matrix_load %A[%row, %k] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %ma1 = air.simdgroup_matrix_load %A[%row1, %k] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %ma2 = air.simdgroup_matrix_load %A[%row2, %k] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %ma3 = air.simdgroup_matrix_load %A[%row3, %k] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %mb0 = air.simdgroup_matrix_load %B[%k, %col] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %mb1 = air.simdgroup_matrix_load %B[%k, %col1] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %mb2 = air.simdgroup_matrix_load %B[%k, %col2] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %mb3 = air.simdgroup_matrix_load %B[%k, %col3] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
        %n00 = air.simdgroup_matrix_multiply_accumulate %ma0, %mb0, %a00 : vector<64xf32>
        %n01 = air.simdgroup_matrix_multiply_accumulate %ma0, %mb1, %a01 : vector<64xf32>
        %n02 = air.simdgroup_matrix_multiply_accumulate %ma0, %mb2, %a02 : vector<64xf32>
        %n03 = air.simdgroup_matrix_multiply_accumulate %ma0, %mb3, %a03 : vector<64xf32>
        %n10 = air.simdgroup_matrix_multiply_accumulate %ma1, %mb0, %a10 : vector<64xf32>
        %n11 = air.simdgroup_matrix_multiply_accumulate %ma1, %mb1, %a11 : vector<64xf32>
        %n12 = air.simdgroup_matrix_multiply_accumulate %ma1, %mb2, %a12 : vector<64xf32>
        %n13 = air.simdgroup_matrix_multiply_accumulate %ma1, %mb3, %a13 : vector<64xf32>
        %n20 = air.simdgroup_matrix_multiply_accumulate %ma2, %mb0, %a20 : vector<64xf32>
        %n21 = air.simdgroup_matrix_multiply_accumulate %ma2, %mb1, %a21 : vector<64xf32>
        %n22 = air.simdgroup_matrix_multiply_accumulate %ma2, %mb2, %a22 : vector<64xf32>
        %n23 = air.simdgroup_matrix_multiply_accumulate %ma2, %mb3, %a23 : vector<64xf32>
        %n30 = air.simdgroup_matrix_multiply_accumulate %ma3, %mb0, %a30 : vector<64xf32>
        %n31 = air.simdgroup_matrix_multiply_accumulate %ma3, %mb1, %a31 : vector<64xf32>
        %n32 = air.simdgroup_matrix_multiply_accumulate %ma3, %mb2, %a32 : vector<64xf32>
        %n33 = air.simdgroup_matrix_multiply_accumulate %ma3, %mb3, %a33 : vector<64xf32>
        scf.yield %n00, %n01, %n02, %n03, %n10, %n11, %n12, %n13, %n20, %n21, %n22, %n23, %n30, %n31, %n32, %n33 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
      }
      scf.yield %in#0, %in#1, %in#2, %in#3, %in#4, %in#5, %in#6, %in#7, %in#8, %in#9, %in#10, %in#11, %in#12, %in#13, %in#14, %in#15 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
    }
    air.simdgroup_matrix_store %acc#0, %C[%row, %col] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#1, %C[%row, %col1] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#2, %C[%row, %col2] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#3, %C[%row, %col3] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#4, %C[%row1, %col] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#5, %C[%row1, %col1] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#6, %C[%row1, %col2] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#7, %C[%row1, %col3] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#8, %C[%row2, %col] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#9, %C[%row2, %col1] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#10, %C[%row2, %col2] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#11, %C[%row2, %col3] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#12, %C[%row3, %col] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#13, %C[%row3, %col1] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#14, %C[%row3, %col2] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    air.simdgroup_matrix_store %acc#15, %C[%row3, %col3] : vector<64xf32> -> memref<?x?xf32, strided<[?, 1]>, 1>
    gpu.return
  }
}
