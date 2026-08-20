// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL:*
// Round trip test for xevm.truncf followed by xevm.extf with the fp4 (e2m1)
// format, at the SPIR-V vector lengths other than 16, which
// xevm_truncf_extf_roundtrip_fp4.mlir covers.
//
// Each of the 16 lanes owns one row of 32 f16 values and converts four slices of
// it, of 2, 3, 4 and 8 elements. Every value used is exactly representable in
// e2m1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), so each round trip must reproduce its
// input. The packed widths differ per slice: 2 values pack into a single byte,
// which SPIR-V spells as a scalar, 3 and 4 into vector<2xi8>, and 8 into
// vector<4xi8>.
//
// Each slice starts at a 16 byte aligned offset in the row, at elements 0, 8, 16
// and 24, so the gaps between them are never written. Results go to a second
// buffer pre-filled with -1, so the gaps read back as -1 and a conversion that
// wrote nothing would leave -1 where a value is expected.
module @roundtrip attributes {gpu.container_module} {

  gpu.module @kernel {
    gpu.func @roundtrip_fp4_lengths(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) kernel {
      %lane = gpu.lane_id
      %lane_i64 = arith.index_cast %lane : index to i64
      %row_len = arith.constant 32 : i64
      %row = arith.muli %lane_i64, %row_len : i64
      %src_row = llvm.getelementptr %src[%row]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %dst_row = llvm.getelementptr %dst[%row]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16

      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c24 = arith.constant 24 : i64

      // 2 elements, packed into one byte, so the packed value is a scalar.
      %v2 = llvm.load %src_row : !llvm.ptr<1> -> vector<2xf16>
      %t2 = xevm.truncf %v2 { src_etype = f16, dst_etype = e2m1 }
          : (vector<2xf16>) -> i8
      %e2 = xevm.extf %t2 { src_etype = e2m1, dst_etype = f16 }
          : (i8) -> vector<2xf16>
      llvm.store %e2, %dst_row : vector<2xf16>, !llvm.ptr<1>

      // 3 elements, padded up to a whole pair, leaving one spare nibble.
      %src3 = llvm.getelementptr %src_row[%c8]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %dst3 = llvm.getelementptr %dst_row[%c8]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %v3 = llvm.load %src3 : !llvm.ptr<1> -> vector<3xf16>
      %t3 = xevm.truncf %v3 { src_etype = f16, dst_etype = e2m1 }
          : (vector<3xf16>) -> vector<2xi8>
      %e3 = xevm.extf %t3 { src_etype = e2m1, dst_etype = f16 }
          : (vector<2xi8>) -> vector<3xf16>
      llvm.store %e3, %dst3 : vector<3xf16>, !llvm.ptr<1>

      // 4 elements, one call whose two written bytes are compacted.
      %src4 = llvm.getelementptr %src_row[%c16]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %dst4 = llvm.getelementptr %dst_row[%c16]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %v4 = llvm.load %src4 : !llvm.ptr<1> -> vector<4xf16>
      %t4 = xevm.truncf %v4 { src_etype = f16, dst_etype = e2m1 }
          : (vector<4xf16>) -> vector<2xi8>
      %e4 = xevm.extf %t4 { src_etype = e2m1, dst_etype = f16 }
          : (vector<2xi8>) -> vector<4xf16>
      llvm.store %e4, %dst4 : vector<4xf16>, !llvm.ptr<1>

      // 8 elements, the point at which two calls fill a whole dword.
      %src8 = llvm.getelementptr %src_row[%c24]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %dst8 = llvm.getelementptr %dst_row[%c24]
          : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %v8 = llvm.load %src8 : !llvm.ptr<1> -> vector<8xf16>
      %t8 = xevm.truncf %v8 { src_etype = f16, dst_etype = e2m1 }
          : (vector<8xf16>) -> vector<4xi8>
      %e8 = xevm.extf %t8 { src_etype = e2m1, dst_etype = f16 }
          : (vector<4xi8>) -> vector<8xf16>
      llvm.store %e8, %dst8 : vector<8xf16>, !llvm.ptr<1>

      gpu.return
    }
  }

  func.func @test(%src : memref<16x32xf16>, %dst : memref<16x32xf16>) -> memref<16x32xf16>
      attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %dev_src = gpu.alloc() : memref<16x32xf16>
    %dev_dst = gpu.alloc() : memref<16x32xf16>
    gpu.memcpy %dev_src, %src : memref<16x32xf16>, memref<16x32xf16>
    gpu.memcpy %dev_dst, %dst : memref<16x32xf16>, memref<16x32xf16>
    %s0 = memref.extract_aligned_pointer_as_index %dev_src : memref<16x32xf16> -> index
    %s1 = arith.index_cast %s0 : index to i64
    %s2 = llvm.inttoptr %s1 : i64 to !llvm.ptr
    %src_casted = llvm.addrspacecast %s2 : !llvm.ptr to !llvm.ptr<1>
    %d0 = memref.extract_aligned_pointer_as_index %dev_dst : memref<16x32xf16> -> index
    %d1 = arith.index_cast %d0 : index to i64
    %d2 = llvm.inttoptr %d1 : i64 to !llvm.ptr
    %dst_casted = llvm.addrspacecast %d2 : !llvm.ptr to !llvm.ptr<1>
    gpu.launch_func @kernel::@roundtrip_fp4_lengths blocks in (%c1, %c1, %c1)
        threads in (%c16, %c1, %c1)
        args(%src_casted : !llvm.ptr<1>, %dst_casted : !llvm.ptr<1>)
    %out = memref.alloc() : memref<16x32xf16>
    gpu.memcpy %out, %dev_dst : memref<16x32xf16>, memref<16x32xf16>
    gpu.dealloc %dev_src : memref<16x32xf16>
    gpu.dealloc %dev_dst : memref<16x32xf16>
    return %out : memref<16x32xf16>
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
    %c32 = arith.constant 32 : index

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

    // Source rows repeat the representable values, so each slice starts at 0 and
    // the 8 element slice covers the whole set.
    %A = memref.alloc() : memref<16x32xf16>
    %B = memref.alloc() : memref<16x32xf16>
    %sentinel = arith.constant -1.0 : f16
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %jm8 = arith.remui %j, %c8 : index
        %val = memref.load %lut[%jm8] : memref<8xf16>
        memref.store %val, %A[%i, %j] : memref<16x32xf16>
        memref.store %sentinel, %B[%i, %j] : memref<16x32xf16>
      }
    }

    %C = call @test(%A, %B) : (memref<16x32xf16>, memref<16x32xf16>) -> memref<16x32xf16>

    // Convert the f16 result to f32 so it can be printed with printMemrefF32.
    %Cf32 = memref.alloc() : memref<16x32xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %v = memref.load %C[%i, %j] : memref<16x32xf16>
        %vf = arith.extf %v : f16 to f32
        memref.store %vf, %Cf32[%i, %j] : memref<16x32xf32>
      }
    }
    %C_cast = memref.cast %Cf32 : memref<16x32xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // The four converted slices sit at elements 0, 8, 16 and 24. Everything else
    // keeps the -1 the destination was filled with.
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-16: [0,   0.5,   -1,   -1,   -1,   -1,   -1,   -1,   0,   0.5,   1,   -1,   -1,   -1,   -1,   -1,   0,   0.5,   1,   1.5,   -1,   -1,   -1,   -1,   0,   0.5,   1,   1.5,   2,   3,   4,   6]
    memref.dealloc %A : memref<16x32xf16>
    memref.dealloc %B : memref<16x32xf16>
    memref.dealloc %C : memref<16x32xf16>
    memref.dealloc %Cf32 : memref<16x32xf32>
    memref.dealloc %lut : memref<8xf16>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
