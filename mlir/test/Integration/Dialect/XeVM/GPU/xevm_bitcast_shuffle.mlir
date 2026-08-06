// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// End-to-end test for `xevm.bitcast_shuffle`, which redistributes the bits of
// the source data across the whole sub-group. The operation comes in two forms:
// a pack, taking a vector and returning a scalar, and an unpack, which is its
// inverse. Both kernels pin the sub-group size to 16 with
// `intel_reqd_sub_group_size`, and are launched with 16 threads, so a single
// full sub-group cooperates on the shuffle.
//
// Both kernels read their source data from memory. The operation reinterprets
// the SIMD register layout of the source, so a source that is uniform across
// the sub-group, a splat constant in particular, is not a meaningful input: it
// is held in a scalar register and there is no per-lane layout to reinterpret.
module @bitcast_shuffle attributes {gpu.container_module} {

  gpu.module @kernel {
    // Reversibility check: a pack followed by an unpack back to the original
    // type must reproduce the original data. This holds for any sub-group size,
    // so no assumption is made about the shuffle pattern here.
    // Lane L owns row L of a 16x2 i32 buffer.
    gpu.func @shuffle_roundtrip(%ptr: !llvm.ptr<1>) kernel
        attributes {llvm.intel_reqd_sub_group_size = 16 : i32} {
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %c2 = arith.constant 2 : i64
      %offset = arith.muli %lane_i64, %c2 : i64
      %lane_ptr = llvm.getelementptr %ptr[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      %vec = llvm.load %lane_ptr : !llvm.ptr<1> -> vector<2xi32>
      %packed = xevm.bitcast_shuffle %vec : (vector<2xi32>) -> i64
      %restored = xevm.bitcast_shuffle %packed : (i64) -> vector<2xi32>
      llvm.store %restored, %lane_ptr : vector<2xi32>, !llvm.ptr<1>
      gpu.return
    }

    // Value check of the pack pattern itself, with N = 16 lanes.
    //
    // Every 16-bit unit of the source is tagged with the position it starts out
    // at: lane L holds `[c * 16 + L for c in 0..3]`, so a tag names the
    // (component, lane) pair it comes from. The packed result is split back into
    // 16-bit units and widened to i32, so each printed value names the source
    // unit that ended up there and the whole output is a permutation of 0..63.
    //
    // Number the 16-bit units of the concatenated source stream `u = c * 16 +
    // L`, so a tag is just its own stream position. The result is a scalar, so
    // it has a single component of C = 64 bits, and lane L receives result
    // stream bits `[64L, 64L + 64)`, that is source stream units `4L` through
    // `4L + 3`. The low half of the packed value holds the earliest of them, as
    // the concatenation is little endian.
    gpu.func @shuffle_value(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) kernel
        attributes {llvm.intel_reqd_sub_group_size = 16 : i32} {
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %c4 = arith.constant 4 : i64
      %offset = arith.muli %lane_i64, %c4 : i64
      %src_ptr = llvm.getelementptr %src[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i16
      %vec = llvm.load %src_ptr : !llvm.ptr<1> -> vector<4xi16>
      %res = xevm.bitcast_shuffle %vec : (vector<4xi16>) -> i64
      // Split the packed result back into the 16-bit units it was assembled
      // from and widen them, so that every tag can be read off the output.
      %halves = llvm.bitcast %res : i64 to vector<4xi16>
      %wide = arith.extui %halves : vector<4xi16> to vector<4xi32>
      %dst_ptr = llvm.getelementptr %dst[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %wide, %dst_ptr : vector<4xi32>, !llvm.ptr<1>
      gpu.return
    }
  }

  func.func @test_roundtrip(%src: memref<16x2xi32>) -> memref<16x2xi32>
      attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_0 = gpu.alloc() : memref<16x2xi32>
    gpu.memcpy %memref_0, %src : memref<16x2xi32>, memref<16x2xi32>
    %0 = memref.extract_aligned_pointer_as_index %memref_0
        : memref<16x2xi32> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %casted = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@shuffle_roundtrip
        blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%casted : !llvm.ptr<1>)
    %dst = memref.alloc() : memref<16x2xi32>
    gpu.memcpy %dst, %memref_0 : memref<16x2xi32>, memref<16x2xi32>
    gpu.dealloc %memref_0 : memref<16x2xi32>
    return %dst : memref<16x2xi32>
  }

  func.func @test_shuffle(%src: memref<16x4xi16>) -> memref<16x4xi32>
      attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %src_gpu = gpu.alloc() : memref<16x4xi16>
    gpu.memcpy %src_gpu, %src : memref<16x4xi16>, memref<16x4xi16>
    %dst_gpu = gpu.alloc() : memref<16x4xi32>
    %0 = memref.extract_aligned_pointer_as_index %src_gpu
        : memref<16x4xi16> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %src_ptr = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    %3 = memref.extract_aligned_pointer_as_index %dst_gpu
        : memref<16x4xi32> -> index
    %4 = arith.index_cast %3 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %dst_ptr = llvm.addrspacecast %5 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@shuffle_value
        blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_ptr : !llvm.ptr<1>, %dst_ptr : !llvm.ptr<1>)
    %dst = memref.alloc() : memref<16x4xi32>
    gpu.memcpy %dst, %dst_gpu : memref<16x4xi32>, memref<16x4xi32>
    gpu.dealloc %src_gpu : memref<16x4xi16>
    gpu.dealloc %dst_gpu : memref<16x4xi32>
    return %dst : memref<16x4xi32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i16 = arith.constant 16 : i16

    // Fill the buffer with 1..32 in row-major order, so that every lane holds
    // two distinct values and no value is repeated across the sub-group.
    %A = memref.alloc() : memref<16x2xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %j_i32 = arith.index_cast %j : index to i32
        %row = arith.muli %i_i32, %c2_i32 : i32
        %idx = arith.addi %row, %j_i32 : i32
        %v = arith.addi %idx, %c1_i32 : i32
        memref.store %v, %A[%i, %j] : memref<16x2xi32>
      }
    }

    %B = call @test_roundtrip(%A) : (memref<16x2xi32>) -> memref<16x2xi32>
    %B_cast = memref.cast %B : memref<16x2xi32> to memref<*xi32>
    call @printMemrefI32(%B_cast) : (memref<*xi32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [1,   2]
    // CHECK: [3,   4]
    // CHECK: [5,   6]
    // CHECK: [7,   8]
    // CHECK: [9,   10]
    // CHECK: [11,   12]
    // CHECK: [13,   14]
    // CHECK: [15,   16]
    // CHECK: [17,   18]
    // CHECK: [19,   20]
    // CHECK: [21,   22]
    // CHECK: [23,   24]
    // CHECK: [25,   26]
    // CHECK: [27,   28]
    // CHECK: [29,   30]
    // CHECK: [31,   32]

    // Tag the 16-bit unit held by lane L as component c with its own position
    // in the concatenated source stream, `c * 16 + L`. Row L of the input is
    // therefore [L, 16+L, 32+L, 48+L].
    %C = memref.alloc() : memref<16x4xi16>
    scf.for %l = %c0 to %c16 step %c1 {
      scf.for %c = %c0 to %c4 step %c1 {
        %l_i16 = arith.index_cast %l : index to i16
        %c_i16 = arith.index_cast %c : index to i16
        %col = arith.muli %c_i16, %c16_i16 : i16
        %tag = arith.addi %col, %l_i16 : i16
        memref.store %tag, %C[%l, %c] : memref<16x4xi16>
      }
    }

    %D = call @test_shuffle(%C) : (memref<16x4xi16>) -> memref<16x4xi32>
    %D_cast = memref.cast %D : memref<16x4xi32> to memref<*xi32>
    call @printMemrefI32(%D_cast) : (memref<*xi32>) -> ()

    // The packed result of lane L holds source stream units 4L through 4L + 3,
    // so the output is the source stream laid out contiguously per lane.
    //
    // A result that reproduces the input rows instead, that is row L reading
    // [L, 16+L, 32+L, 48+L], means no data crossed lanes and the pack
    // degenerated into a per-lane bitcast.
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [0,   1,   2,   3]
    // CHECK: [4,   5,   6,   7]
    // CHECK: [8,   9,   10,   11]
    // CHECK: [12,   13,   14,   15]
    // CHECK: [16,   17,   18,   19]
    // CHECK: [20,   21,   22,   23]
    // CHECK: [24,   25,   26,   27]
    // CHECK: [28,   29,   30,   31]
    // CHECK: [32,   33,   34,   35]
    // CHECK: [36,   37,   38,   39]
    // CHECK: [40,   41,   42,   43]
    // CHECK: [44,   45,   46,   47]
    // CHECK: [48,   49,   50,   51]
    // CHECK: [52,   53,   54,   55]
    // CHECK: [56,   57,   58,   59]
    // CHECK: [60,   61,   62,   63]

    memref.dealloc %A : memref<16x2xi32>
    memref.dealloc %B : memref<16x2xi32>
    memref.dealloc %C : memref<16x4xi16>
    memref.dealloc %D : memref<16x4xi32>
    return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi32>) attributes { llvm.emit_c_interface }
}
