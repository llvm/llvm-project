// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL:*
// Value check (not just a round trip) for xevm.truncf / xevm.extf.
// Each of the 16 lanes owns a vector<16xf16> initialized to (lane + 1); the
// integers 1..16 are all exactly representable in f8E4M3FN. xevm.truncf
// down-converts to f8 and the raw f8 bytes are written out so the
// down-converted result can be checked against the known E4M3FN bit patterns.
// xevm.extf then up-converts the f8 bytes back to f16 so the up-converted
// result can be checked as well.
module @raw_convert attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @truncf_extf(%f16_ptr: !llvm.ptr<1>, %f8_ptr: !llvm.ptr<1>) kernel {
      // Lane L owns f16/f8 elements [L*16, L*16+16).
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %c16 = arith.constant 16 : i64
      %offset = arith.muli %lane_i64, %c16 : i64
      %f16_lane_ptr = llvm.getelementptr %f16_ptr[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %f8_lane_ptr = llvm.getelementptr %f8_ptr[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
      %vec = llvm.load %f16_lane_ptr : !llvm.ptr<1> -> vector<16xf16>
      // Down-convert f16 -> f8 (E4M3FN); store the raw f8 bytes.
      %trunc = xevm.truncf %vec { src_etype = f16, dst_etype = f8 }
          : (vector<16xf16>) -> vector<16xi8>
      llvm.store %trunc, %f8_lane_ptr : vector<16xi8>, !llvm.ptr<1>
      // Up-convert f8 -> f16; store the result back over the input.
      %ext = xevm.extf %trunc { src_etype = f8, dst_etype = f16 }
          : (vector<16xi8>) -> vector<16xf16>
      llvm.store %ext, %f16_lane_ptr : vector<16xf16>, !llvm.ptr<1>
      gpu.return
    }
  }

  // Returns the up-converted f16 result and the raw down-converted f8 bytes.
  func.func @test(%src : memref<16x16xf16>) -> (memref<16x16xf16>, memref<16x16xi8>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %f16_dev = gpu.alloc() : memref<16x16xf16>
    gpu.memcpy %f16_dev, %src : memref<16x16xf16>, memref<16x16xf16>
    %f8_dev = gpu.alloc() : memref<16x16xi8>
    %f16_idx = memref.extract_aligned_pointer_as_index %f16_dev : memref<16x16xf16> -> index
    %f16_i64 = arith.index_cast %f16_idx : index to i64
    %f16_llvm = llvm.inttoptr %f16_i64 : i64 to !llvm.ptr
    %f16_casted = llvm.addrspacecast %f16_llvm : !llvm.ptr to !llvm.ptr<1>
    %f8_idx = memref.extract_aligned_pointer_as_index %f8_dev : memref<16x16xi8> -> index
    %f8_i64 = arith.index_cast %f8_idx : index to i64
    %f8_llvm = llvm.inttoptr %f8_i64 : i64 to !llvm.ptr
    %f8_casted = llvm.addrspacecast %f8_llvm : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@truncf_extf blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%f16_casted : !llvm.ptr<1>, %f8_casted : !llvm.ptr<1>)
    %f16_host = memref.alloc() : memref<16x16xf16>
    gpu.memcpy %f16_host, %f16_dev : memref<16x16xf16>, memref<16x16xf16>
    %f8_host = memref.alloc() : memref<16x16xi8>
    gpu.memcpy %f8_host, %f8_dev : memref<16x16xi8>, memref<16x16xi8>
    gpu.dealloc %f16_dev : memref<16x16xf16>
    gpu.dealloc %f8_dev : memref<16x16xi8>
    return %f16_host, %f8_host : memref<16x16xf16>, memref<16x16xi8>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32

    // Fill row L with the value (L + 1), exactly representable in f8E4M3FN.
    %A = memref.alloc() : memref<16x16xf16>
    scf.for %i = %c0 to %c16 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %v_i32 = arith.addi %i_i32, %c1_i32 : i32
      %v = arith.sitofp %v_i32 : i32 to f16
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %v, %A[%i, %j] : memref<16x16xf16>
      }
    }

    %f16_res, %f8_res = call @test(%A)
        : (memref<16x16xf16>) -> (memref<16x16xf16>, memref<16x16xi8>)

    // Down-converted result: the raw f8 (E4M3FN) bytes. Sign-extend to i32 so
    // they print as integers (printMemrefI8 would render them as characters).
    // For value (L + 1), the expected E4M3FN encodings (row L) are:
    //   1->0x38, 2->0x40, 3->0x44, 4->0x48, 5->0x4a, 6->0x4c, 7->0x4e, 8->0x50,
    //   9->0x51, 10->0x52, 11->0x53, 12->0x54, 13->0x55, 14->0x56, 15->0x57,
    //   16->0x58.
    %raw = memref.alloc() : memref<16x16xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %b = memref.load %f8_res[%i, %j] : memref<16x16xi8>
        %bi = arith.extsi %b : i8 to i32
        memref.store %bi, %raw[%i, %j] : memref<16x16xi32>
      }
    }
    %raw_cast = memref.cast %raw : memref<16x16xi32> to memref<*xi32>
    call @printMemrefI32(%raw_cast) : (memref<*xi32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56,   56]
    // CHECK: [64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64]
    // CHECK: [68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68,   68]
    // CHECK: [72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72,   72]
    // CHECK: [74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74,   74]
    // CHECK: [76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76,   76]
    // CHECK: [78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78,   78]
    // CHECK: [80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80,   80]
    // CHECK: [81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81,   81]
    // CHECK: [82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82,   82]
    // CHECK: [83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83]
    // CHECK: [84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84,   84]
    // CHECK: [85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85,   85]
    // CHECK: [86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86,   86]
    // CHECK: [87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87,   87]
    // CHECK: [88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88,   88]

    // Up-converted result: f8 -> f16, printed as f32. The round trip is lossless
    // for these inputs, so each row reproduces its (L + 1) value.
    %Bf32 = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %f16_res[%i, %j] : memref<16x16xf16>
        %vf = arith.extf %v : f16 to f32
        memref.store %vf, %Bf32[%i, %j] : memref<16x16xf32>
      }
    }
    %B_cast = memref.cast %Bf32 : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
    // CHECK: [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]
    // CHECK: [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3]
    // CHECK: [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4]
    // CHECK: [5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5]
    // CHECK: [6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6]
    // CHECK: [7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7]
    // CHECK: [8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8]
    // CHECK: [9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9]
    // CHECK: [10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10]
    // CHECK: [11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11,   11]
    // CHECK: [12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12,   12]
    // CHECK: [13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13,   13]
    // CHECK: [14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14,   14]
    // CHECK: [15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15,   15]
    // CHECK: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16]

    memref.dealloc %A : memref<16x16xf16>
    memref.dealloc %f16_res : memref<16x16xf16>
    memref.dealloc %f8_res : memref<16x16xi8>
    memref.dealloc %raw : memref<16x16xi32>
    memref.dealloc %Bf32 : memref<16x16xf32>
    return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi32>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
