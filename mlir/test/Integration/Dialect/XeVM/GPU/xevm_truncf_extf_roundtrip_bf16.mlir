// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL:*
// Round trip test for xevm.truncf followed by xevm.extf.
// Each of the 16 lanes owns a vector<16xbf16>, truncates it to f8 (E4M3FN) with
// xevm.truncf and extends it back to bf16 with xevm.extf. The integers 1..16 are
// exactly representable in both f8E4M3FN and bf16, so the round trip must
// reproduce the input.
module @roundtrip attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @truncf_extf_roundtrip(%ptr: !llvm.ptr<1>) kernel {
      // Each lane processes 16 contiguous bf16 values: lane L owns [L*16, L*16+16).
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %c16 = arith.constant 16 : i64
      %offset = arith.muli %lane_i64, %c16 : i64
      %lane_ptr = llvm.getelementptr %ptr[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, bf16
      %vec = llvm.load %lane_ptr : !llvm.ptr<1> -> vector<16xbf16>
      // bf16 -> f8 (E4M3FN) -> bf16 round trip.
      %trunc = xevm.truncf %vec { src_etype = bf16, dst_etype = f8 }
          : (vector<16xbf16>) -> vector<16xi8>
      %ext = xevm.extf %trunc { src_etype = f8, dst_etype = bf16 }
          : (vector<16xi8>) -> vector<16xbf16>
      llvm.store %ext, %lane_ptr : vector<16xbf16>, !llvm.ptr<1>
      gpu.return
    }
  }

  func.func @test(%src : memref<16x16xbf16>) -> memref<16x16xbf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_0 = gpu.alloc() : memref<16x16xbf16>
    gpu.memcpy %memref_0, %src : memref<16x16xbf16>, memref<16x16xbf16>
    %0 = memref.extract_aligned_pointer_as_index %memref_0 : memref<16x16xbf16> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %src_casted = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@truncf_extf_roundtrip blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_casted : !llvm.ptr<1>)
    %dst = memref.alloc() : memref<16x16xbf16>
    gpu.memcpy %dst, %memref_0 : memref<16x16xbf16>, memref<16x16xbf16>
    gpu.dealloc %memref_0 : memref<16x16xbf16>
    return %dst : memref<16x16xbf16>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32

    // Fill row L with the value (L + 1), exactly representable in f8E4M3FN.
    %A = memref.alloc() : memref<16x16xbf16>
    scf.for %i = %c0 to %c16 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %v_i32 = arith.addi %i_i32, %c1_i32 : i32
      %v = arith.sitofp %v_i32 : i32 to bf16
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %v, %A[%i, %j] : memref<16x16xbf16>
      }
    }

    %B = call @test(%A) : (memref<16x16xbf16>) -> memref<16x16xbf16>

    // Convert the bf16 result to f32 so it can be printed with printMemrefF32.
    %Bf32 = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %B[%i, %j] : memref<16x16xbf16>
        %vf = arith.extf %v : bf16 to f32
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
    memref.dealloc %A : memref<16x16xbf16>
    memref.dealloc %B : memref<16x16xbf16>
    memref.dealloc %Bf32 : memref<16x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
