// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Round trip test for xevm.truncf followed by xevm.extf with the fp4 (e2m1)
// format. Each of the 16 lanes owns a vector<16xf16>, truncates it to e2m1 (16
// fp4 values packed into vector<8xi8>) with xevm.truncf and extends it back to
// f16 with xevm.extf. Every value used is exactly representable in e2m1
// (0, 0.5, 1, 1.5, 2, 3, 4, 6), so the round trip must reproduce the input.
module @roundtrip attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @truncf_extf_roundtrip_fp4(%ptr: !llvm.ptr<1>) kernel {
      // Each lane processes 16 contiguous f16 values: lane L owns [L*16, L*16+16).
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %c16 = arith.constant 16 : i64
      %offset = arith.muli %lane_i64, %c16 : i64
      %lane_ptr = llvm.getelementptr %ptr[%offset]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %vec = llvm.load %lane_ptr : !llvm.ptr<1> -> vector<16xf16>
      // f16 -> e2m1 (fp4, 16 values packed in vector<8xi8>) -> f16 round trip.
      %trunc = xevm.truncf %vec { src_etype = f16, dst_etype = e2m1 }
          : (vector<16xf16>) -> vector<8xi8>
      %ext = xevm.extf %trunc { src_etype = e2m1, dst_etype = f16 }
          : (vector<8xi8>) -> vector<16xf16>
      llvm.store %ext, %lane_ptr : vector<16xf16>, !llvm.ptr<1>
      gpu.return
    }
  }

  func.func @test(%src : memref<16x16xf16>) -> memref<16x16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_0 = gpu.alloc() : memref<16x16xf16>
    gpu.memcpy %memref_0, %src : memref<16x16xf16>, memref<16x16xf16>
    %0 = memref.extract_aligned_pointer_as_index %memref_0 : memref<16x16xf16> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %src_casted = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@truncf_extf_roundtrip_fp4 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_casted : !llvm.ptr<1>)
    %dst = memref.alloc() : memref<16x16xf16>
    gpu.memcpy %dst, %memref_0 : memref<16x16xf16>, memref<16x16xf16>
    gpu.dealloc %memref_0 : memref<16x16xf16>
    return %dst : memref<16x16xf16>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    // Lookup table of the 8 magnitudes exactly representable in e2m1.
    %lut = memref.alloc() : memref<8xf16>
    %v0 = arith.constant 0.0 : f16
    %v1 = arith.constant 0.5 : f16
    %v2 = arith.constant 1.0 : f16
    %v3 = arith.constant 1.5 : f16
    %v4 = arith.constant 2.0 : f16
    %v5 = arith.constant 3.0 : f16
    %v6 = arith.constant 4.0 : f16
    %v7 = arith.constant 6.0 : f16
    memref.store %v0, %lut[%c0] : memref<8xf16>
    memref.store %v1, %lut[%c1] : memref<8xf16>
    memref.store %v2, %lut[%c2] : memref<8xf16>
    memref.store %v3, %lut[%c3] : memref<8xf16>
    memref.store %v4, %lut[%c4] : memref<8xf16>
    memref.store %v5, %lut[%c5] : memref<8xf16>
    memref.store %v6, %lut[%c6] : memref<8xf16>
    memref.store %v7, %lut[%c7] : memref<8xf16>

    // Fill every row with the repeating pattern of representable values so each
    // lane exercises the full set of e2m1 values.
    %A = memref.alloc() : memref<16x16xf16>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %jm8 = arith.remui %j, %c8 : index
        %val = memref.load %lut[%jm8] : memref<8xf16>
        memref.store %val, %A[%i, %j] : memref<16x16xf16>
      }
    }

    %B = call @test(%A) : (memref<16x16xf16>) -> memref<16x16xf16>

    // Convert the f16 result to f32 so it can be printed with printMemrefF32.
    %Bf32 = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %B[%i, %j] : memref<16x16xf16>
        %vf = arith.extf %v : f16 to f32
        memref.store %vf, %Bf32[%i, %j] : memref<16x16xf32>
      }
    }
    %B_cast = memref.cast %Bf32 : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-16: [0,   0.5,   1,   1.5,   2,   3,   4,   6,   0,   0.5,   1,   1.5,   2,   3,   4,   6]
    memref.dealloc %A : memref<16x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    memref.dealloc %Bf32 : memref<16x16xf32>
    memref.dealloc %lut : memref<8xf16>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
