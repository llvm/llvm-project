//
// NOTE: this test requires gpu-sm80
//
// RUN: mlir-opt \
// RUN: --pass-pipeline="builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-nvgpu-to-nvvm,affine-expand-index-ops,lower-affine,convert-arith-to-llvm),convert-vector-to-llvm,canonicalize,cse)" \
// RUN: %s \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --e main --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {

  // Kernels that run on the device.

  gpu.module @kernels {

    //
    // An NVidia GPU kernel to compute
    //   C = A x B
    // (or, technically, D = A x B + C)
    // using 2:4 structured sparsity for A.
    //
    // This kernel provides building block for sparse compilation of a larger
    // enveloping matrix multiplication computation on a GPU.
    //
    // Operand A values (2:4 sparse): row major format, logically "16x32xf16"
    //                                but "16x16xf16" after compression
    //
    // Operand A metadata:
    //   - The metadata is logically "16x16xi2". Each 2-bit value indicates
    //     the position of a non-zero value within the respective group of 4 elements.
    //   - However, we represent it as "16x2xi16".
    //   - Each sparse instruction type specifies how the metadata should be distributed
    //     among threads. In this case, within each quad (group of 4 consecutive threads
    //     starting with a thread ID which is a multiple of 4), thread 4i and 4i+1
    //     will require to hold the metadata different metadata. For uniformity below,
    //     we just have all threads load metadata, and the way they determine which metadata
    //     to load is given below.
    //   - Thread map for the 16x32x16 instruction is:
    //                     2i -> col 0
    //                 2i + 1 -> col 1
    //
    // Operand B (dense): column major format.
    //
    // Operand C (accum): assumed zero on entry, used as output.
    //
    gpu.func @mma_sp_sync_f16_16832(
        %argA: memref<16x16xf16>,
        %argA_meta: memref<16x2xi16>,
        %argB: memref<8x32xf16>,
        %argC: memref<16x8xf16>) kernel {
      %f0 = arith.constant 0.0 : f16
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index

      // Assume we have a linear thread id and the kernel launches 32 threads (1 warp).
      // So CUDA launch would be threadblock = (32, 1, 1), grid = (1, 1, 1)
      %lane_id = gpu.thread_id x
      // Which group of 4 threads do we belong to?
      %quad_id = affine.apply affine_map<()[s0]->(s0 floordiv 4)>()[%lane_id]
      // Are we even group or odd group?
      %pair_id = affine.apply affine_map<()[s0]->(s0 mod 2)>()[%lane_id]

      // Now we have
      // MMA lane=0 quad=0 pair=0
      // MMA lane=1 quad=0 pair=1
      // MMA lane=2 quad=0 pair=0
      // MMA lane=3 quad=0 pair=1
      // MMA lane=4 quad=1 pair=0
      // MMA lane=5 quad=1 pair=1
      // ...
      // MMA lane=30 quad=7 pair=2
      // MMA lane=31 quad=7 pair=1
      //
      // gpu.printf "MMA lane=%lld quad=%lld pair=%lld\n" %lane_id, %quad_id, %pair_id : index, index, index

      //===----------------------------------------------------------------------===//
      // Load the operandA metadata
      // (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-mma-metadata-16832-f16bf16)
      //===----------------------------------------------------------------------===//

      // For the 16x2xi16 metadata, all threads that load metadata will load one
      // i16 value from the first 8 rows, and one i16 value from the second 8 rows.
      // The i16 values are then put into a i32 with the value from the first 8 rows
      // going in the lower bits.
      //
      // The below IR loads and combines the two pieces of i16 metadata required.
      // Obviously, it's possible to re-pack the metadata before launching the kernel in
      // order to eliminate this cost and load a single i32 operand. This just shows
      // how to put them together if you do the naive load per the diagram in
      // the PTX docs. Technically only the first two threads in each quad need
      // to do this, but for simplicity we just have all threads participate since
      // it can't hurt.
      //
      // The mapping is
      // Lower i16 bits <- (thread_id) -> load A_meta[ quad_id    , pair_id]
      // Lower i16 bits <- (thread_id) -> load A_meta[ quad_id + 8, pair_id]

      %quad_id_plus_8 = affine.apply affine_map<()[s0]->(s0 + 8)>()[%quad_id]
      %meta_A_per_thread0 = memref.load %argA_meta[%quad_id       , %pair_id] : memref<16x2xi16>
      %meta_A_per_thread1 = memref.load %argA_meta[%quad_id_plus_8, %pair_id] : memref<16x2xi16>

      %low_i32  = arith.extui %meta_A_per_thread0 : i16 to i32
      %high_i32 = arith.extui %meta_A_per_thread1 : i16 to i32

      %meta_init = arith.constant dense<0> : vector<2xi16>
      %meta_low  = vector.insert %meta_A_per_thread0, %meta_init[0] : i16 into vector<2xi16>
      %meta      = vector.insert %meta_A_per_thread1, %meta_low[1]  : i16 into vector<2xi16>

      //===----------------------------------------------------------------------===//
      // Load operandA
      //===----------------------------------------------------------------------===//

      // Load the actual fragments for the sparse values. This can be done using ldmatrix,
      // but here we just do naive individual loads, which would also be required for
      // a layout/element type that is not compatible with ldmatrix (e.g. i8 transpose load).
      //
      // The thread map here is different that operandA metadata. Each thread will
      // load one 2xf16 vector from each of the four (8x8xf16) quadrants fo the 16x16xf16
      // operand.
      //
      // The (thread_id)->(row, col) map within each 8x4x(2xf16) quadrant is (t)->(t/4, t%4). We
      // can use "affine.delinearize_index" which means the same thing.

      %quad_row, %col_8x4 = affine.delinearize_index %lane_id into (%c8, %c4) : index, index
      %quad_col = affine.apply affine_map<()[s0]->(s0 * 2)>()[%col_8x4] // account for 2xf16/col

      // Load quad (0, 0)
      %A_quad00 = vector.transfer_read %argA[%quad_row, %quad_col], %f0 {in_bounds = [true]} : memref<16x16xf16>, vector<2xf16>

      // Load quad (1, 0). Just shift row down 8.
      %quad_row_plus_8 = affine.apply affine_map<(d0)[]->(d0+8)>(%quad_row)[]
      %A_quad10 = vector.transfer_read %argA[%quad_row_plus_8, %quad_col], %f0 {in_bounds = [true]} : memref<16x16xf16>, vector<2xf16>

      // Load quad (0, 1). Just shift col right 8 (4 2xf16 values)
      %quad_col_plus_8 = affine.apply affine_map<(d0)[]->(d0+8)>(%quad_col)[]
      %A_quad01 = vector.transfer_read %argA[%quad_row, %quad_col_plus_8], %f0 {in_bounds = [true]} : memref<16x16xf16>, vector<2xf16>

      // Load quad (1, 1)
      %A_quad11 = vector.transfer_read %argA[%quad_row_plus_8, %quad_col_plus_8], %f0 {in_bounds = [true]} : memref<16x16xf16>, vector<2xf16>

      // Assemble the elements into a vector
      %A_init0 = arith.constant dense<0.0> : vector<4x2xf16>
      %A_data0 = vector.insert %A_quad00, %A_init0[0] : vector<2xf16> into vector<4x2xf16>
      %A_data1 = vector.insert %A_quad10, %A_data0[1] : vector<2xf16> into vector<4x2xf16>
      %A_data2 = vector.insert %A_quad01, %A_data1[2] : vector<2xf16> into vector<4x2xf16>
      %A_data  = vector.insert %A_quad11, %A_data2[3] : vector<2xf16> into vector<4x2xf16>

      //===----------------------------------------------------------------------===//
      // Load operand B
      //===----------------------------------------------------------------------===//

      // Load the actual fragments for the dense values. This can be done using ldmatrix,
      // but here we just do naive individual loads, as would be required if we could
      // not use ldmatrix.
      //
      // The thread map here is different from operandA. This operand is in the form
      // memref<8x32xf16> (col major). Each thread load a 2xf16 vector from a
      // 8x8xf16 quadrant.
      //
      // The (thread_id)->(col, row) map within each 8x4x(2xf16) quadrant is
      // (t) -> (t/4, t % 4). So we can re-use some of the calculation from A.

      // Load quad (0, 0)
      %B_quad0 = vector.transfer_read %argB[%quad_row, %quad_col],  %f0 {in_bounds = [true]} : memref<8x32xf16>, vector<2xf16>

      // Load quad (0, 1)
      %B_quad1 = vector.transfer_read %argB[%quad_row, %quad_col_plus_8],  %f0 {in_bounds = [true]} : memref<8x32xf16>, vector<2xf16>

      // Load quad (0, 2)
      %quad_col_plus_16 = affine.apply affine_map<()[s0]->(s0 + 16)>()[%quad_col]
      %B_quad2 = vector.transfer_read %argB[%quad_row, %quad_col_plus_16], %f0 {in_bounds = [true]} : memref<8x32xf16>, vector<2xf16>

      // Load quad (0, 3)
      %quad_col_plus_24 = affine.apply affine_map<()[s0]->(s0 + 24)>()[%quad_col]
      %B_quad3 = vector.transfer_read %argB[%quad_row, %quad_col_plus_24], %f0 {in_bounds = [true]} : memref<8x32xf16>, vector<2xf16>

      // Assemble into vector
      %B_init0 = arith.constant dense<0.0> : vector<4x2xf16>
      %B_data0 = vector.insert %B_quad0, %B_init0[0] : vector<2xf16> into vector<4x2xf16>
      %B_data1 = vector.insert %B_quad1, %B_data0[1] : vector<2xf16> into vector<4x2xf16>
      %B_data2 = vector.insert %B_quad2, %B_data1[2] : vector<2xf16> into vector<4x2xf16>
      %B_data  = vector.insert %B_quad3, %B_data2[3] : vector<2xf16> into vector<4x2xf16>

      // For now just say accum is a zero-d register
      %accum = arith.constant dense<0.0> : vector<2x2xf16>

      gpu.barrier

      // Sparsity selector. For 16x8x32, the default "0" means threads T0/T1
      // within each group of four threads contribute metadata.
      %d = nvgpu.mma.sp.sync(%A_data, %B_data, %accum)
           metadata(%meta)
           {mmaShape = [16, 8, 32]} : (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>

      //===----------------------------------------------------------------------===//
      // Write back results to gpu global memory
      //===----------------------------------------------------------------------===//

      // The mma instruction gave us two 2xf16 vectors per thread. These values
      // correspond to different positions in the 16x8xf16 result matrix. Each value belongs
      // to one of the 8x4x(2xf16) halves. The halves are indexed as follows (as you might guess):
      // vector0: (tid) -> (tid / 4    ,  tid %4)
      // vector1: (tid) -> (tid / 4 + 8,  tid %4)
      %C_0 = vector.extract %d[0] : vector<2xf16> from vector<2x2xf16>
      %C_1 = vector.extract %d[1] : vector<2xf16> from vector<2x2xf16>
      vector.transfer_write %C_0, %argC[%quad_row,        %quad_col] {in_bounds = [true]} : vector<2xf16>, memref<16x8xf16>
      vector.transfer_write %C_1, %argC[%quad_row_plus_8, %quad_col] {in_bounds = [true]} : vector<2xf16>, memref<16x8xf16>

      gpu.return
    }
  }

  // Code than runs on the host.

  //
  // This test performs a matrix multiplication
  //   C = A x B
  // using NVidia 2:4 structured sparsity for A.
  //
  func.func @main() {
    %f0  = arith.constant 0.0 : f16
    %c0  = arith.constant 0   : index
    %c1  = arith.constant 1   : index
    %c2  = arith.constant 2   : index
    %c8  = arith.constant 8   : index
    %c16 = arith.constant 16  : index
    %c32 = arith.constant 32  : index
    %c64 = arith.constant 64  : index

    // Matrices A, B, C (16x32, 32x8, 16x8).
    %a = memref.alloc() : memref<16x16xf16>  // 16x32 but 2:4, row-major
    %b = memref.alloc() : memref<8x32xf16>   // regular dense  column-major
    %c = memref.alloc() : memref<16x8xf16>   // accumulator    row-major

    // Metadata for A.
    %m = memref.alloc() : memref<16x2xi16>

    //
    // Setup matrix A.
    //
    scf.for %ai = %c0 to %c16 step %c1 {
      scf.for %aj = %c0 to %c16 step %c1 {
        %a0 = arith.addi %ai, %aj : index
        %a1 = arith.addi %a0, %c1 : index
        %a2 = arith.index_cast %a1 : index to i32
        %a3 = arith.sitofp %a2 : i32 to f16
        memref.store %a3, %a[%ai, %aj] : memref<16x16xf16>
      }
    }

    //
    // Setup metadata for matrix A.
    //
    // Here we assume that all 2:4 elements are in pos 0 and 2,
    // viz. in matrix
    //   | A 0 B 0 |
    //   { 0   2   }
    //
    // Note that within each i16, we need little-endian
    // storage of the indices, as follows:
    //
    //   10 00 10 00 10 00 10 00 10 00 10 00 = 0x8888
    //
    %bits = arith.constant 0x8888 : i16
    scf.for %mi = %c0 to %c16 step %c1 {
      memref.store %bits, %m[%mi, %c0] : memref<16x2xi16>
      memref.store %bits, %m[%mi, %c1] : memref<16x2xi16>
    }

    //
    // Setup matrix B.
    //
    scf.for %bi = %c0 to %c8 step %c1 {
      scf.for %bj = %c0 to %c32 step %c1 {
        %b0 = arith.subi %bi, %bj : index
        %b1 = arith.index_cast %b0 : index to i32
        %b2 = arith.sitofp %b1 : i32 to f16
        memref.store %b2, %b[%bi, %bj] : memref<8x32xf16>
      }
    }

    //
    // Reset matrix C.
    //
    scf.for %ci = %c0 to %c16 step %c1 {
      scf.for %cj = %c0 to %c8 step %c1 {
        memref.store %f0, %c[%ci, %cj] : memref<16x8xf16>
      }
    }

    //
    // Sanity check on **compressed** input matrix A.
    //
    // Note that it really is a 16x32 matrix:
    //   | 1 0 2 0 3 0 ...
    //   | 2 0 3 0 4 0 ...
    //   etc.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 )
    // CHECK-NEXT: ( 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 )
    // CHECK-NEXT: ( 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 )
    // CHECK-NEXT: ( 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 )
    // CHECK-NEXT: ( 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 )
    // CHECK-NEXT: ( 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 )
    // CHECK-NEXT: ( 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 )
    // CHECK-NEXT: ( 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 )
    // CHECK-NEXT: ( 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 )
    // CHECK-NEXT: ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 )
    // CHECK-NEXT: ( 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 )
    // CHECK-NEXT: ( 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 )
    // CHECK-NEXT: ( 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 )
    // CHECK-NEXT: ( 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 )
    // CHECK-NEXT: ( 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 )
    //
    scf.for %pai = %c0 to %c16 step %c1 {
      %pa0 = vector.transfer_read %a[%pai, %c0], %f0 : memref<16x16xf16>, vector<16xf16>
      vector.print %pa0 : vector<16xf16>
    }

    //
    // Sanity check on input matrix 32x8 B.
    // Note that this is really shown as B^T
    //
    // CHECK-NEXT: ( 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31 )
    // CHECK-NEXT: ( 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30 )
    // CHECK-NEXT: ( 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29 )
    // CHECK-NEXT: ( 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28 )
    // CHECK-NEXT: ( 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27 )
    // CHECK-NEXT: ( 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26 )
    // CHECK-NEXT: ( 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25 )
    // CHECK-NEXT: ( 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24 )
    //
    //
    scf.for %pbi = %c0 to %c8 step %c1 {
      %pb0 = vector.transfer_read %b[%pbi, %c0], %f0 : memref<8x32xf16>, vector<32xf16>
      vector.print %pb0 : vector<32xf16>
    }

    // Maps the provided host buffers into the device address space.
    // Writes from the host are guaranteed to be visible to device
    // kernels that are launched afterwards. Writes from the device
    // are guaranteed to be visible on the host after synchronizing
    // with the device kernel completion.
    %cast_a = memref.cast %a : memref<16x16xf16> to memref<*xf16>
    gpu.host_register %cast_a : memref<*xf16>
    %cast_m = memref.cast %m : memref<16x2xi16> to memref<*xi16>
    gpu.host_register %cast_m : memref<*xi16>
    %cast_b = memref.cast %b : memref<8x32xf16> to memref<*xf16>
    gpu.host_register %cast_b : memref<*xf16>
    %cast_c = memref.cast %c : memref<16x8xf16> to memref<*xf16>
    gpu.host_register %cast_c : memref<*xf16>

    // Call the kernel, using a single warp of 32 threads.
    %t1  = arith.constant 1  : index
    %t32 = arith.constant 32 : index
    gpu.launch_func
            @kernels::@mma_sp_sync_f16_16832
            blocks  in (%t1,  %t1, %t1) // gridSizeX,Y,Z
            threads in (%t32, %t1, %t1) // blockSizeX,Y,Z
            args(%a : memref<16x16xf16>,
                 %m : memref<16x2xi16>,
                 %b : memref<8x32xf16>,
                 %c : memref<16x8xf16>)

    // Unmaps the host buffers.
    gpu.host_unregister %cast_a : memref<*xf16>
    gpu.host_unregister %cast_m : memref<*xi16>
    gpu.host_unregister %cast_b : memref<*xf16>
    gpu.host_unregister %cast_c : memref<*xf16>

    //
    // Verify computed matrix C.
    //
    // CHECK-NEXT: ( -2720, -2584, -2448, -2312, -2176, -2040, -1904, -1768 )
    // CHECK-NEXT: ( -2960, -2808, -2656, -2504, -2352, -2200, -2048, -1896 )
    // CHECK-NEXT: ( -3200, -3032, -2864, -2696, -2528, -2360, -2192, -2024 )
    // CHECK-NEXT: ( -3440, -3256, -3072, -2888, -2704, -2520, -2336, -2152 )
    // CHECK-NEXT: ( -3680, -3480, -3280, -3080, -2880, -2680, -2480, -2280 )
    // CHECK-NEXT: ( -3920, -3704, -3488, -3272, -3056, -2840, -2624, -2408 )
    // CHECK-NEXT: ( -4160, -3928, -3696, -3464, -3232, -3000, -2768, -2536 )
    // CHECK-NEXT: ( -4400, -4152, -3904, -3656, -3408, -3160, -2912, -2664 )
    // CHECK-NEXT: ( -4640, -4376, -4112, -3848, -3584, -3320, -3056, -2792 )
    // CHECK-NEXT: ( -4880, -4600, -4320, -4040, -3760, -3480, -3200, -2920 )
    // CHECK-NEXT: ( -5120, -4824, -4528, -4232, -3936, -3640, -3344, -3048 )
    // CHECK-NEXT: ( -5360, -5048, -4736, -4424, -4112, -3800, -3488, -3176 )
    // CHECK-NEXT: ( -5600, -5272, -4944, -4616, -4288, -3960, -3632, -3304 )
    // CHECK-NEXT: ( -5840, -5496, -5152, -4808, -4464, -4120, -3776, -3432 )
    // CHECK-NEXT: ( -6080, -5720, -5360, -5000, -4640, -4280, -3920, -3560 )
    // CHECK-NEXT: ( -6320, -5944, -5568, -5192, -4816, -4440, -4064, -3688 )
    //
    scf.for %pci = %c0 to %c16 step %c1 {
      %pc0 = vector.transfer_read %c[%pci, %c0], %f0 : memref<16x8xf16>, vector<8xf16>
      vector.print %pc0 : vector<8xf16>
    }

    return
  }
}
