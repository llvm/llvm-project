// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @dpas_mx_e2m1(%a: memref<8x64xf4E2M1FN>, %b: memref<32x16xi8>, %c: memref<8x16xf32>, %scale_a: memref<8x2xf8E8M0FNU>, %scale_b: memref<2x16xf8E8M0FNU>) kernel {

      %tdesc_a = xegpu.create_nd_tdesc %a : memref<8x64xf4E2M1FN> -> !xegpu.tensor_desc<8x64xf4E2M1FN>
      %a_trunc = xegpu.load_nd %tdesc_a[0, 0] : !xegpu.tensor_desc<8x64xf4E2M1FN> -> vector<32xf4E2M1FN>

      %tdesc_b = xegpu.create_nd_tdesc %b : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8>
      %b_loaded = xegpu.load_nd %tdesc_b[0, 0] <{packed}> : !xegpu.tensor_desc<32x16xi8> -> vector<32xi8>
      %b_trunc = vector.bitcast %b_loaded : vector<32xi8> to vector<64xf4E2M1FN>

      %tdesc_c = xegpu.create_nd_tdesc %c : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c_loaded = xegpu.load_nd %tdesc_c[0, 0] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

      // How to represent loaded scale_a in lane level?
      // Not enough columns for scale_a to lane distribute.
      // Option 1: Use load_nd
      // load_nd of tile size MxK where, K is too small and element type is f8E8M0FNU,
      // has a special sematics to load K element vector per lane for the first M lanes,
      // and fill 0 for the rest lanes. This is to support the common usage of scale_a in DPAS MX
      // which has M scale factor vectors for M rows of A.
      //%tdesc_scale_a = xegpu.create_nd_tdesc %scale_a : memref<8x2xf8E8M0FNU> -> !xegpu.tensor_desc<8x2xf8E8M0FNU>
      //%scale_a_loaded = xegpu.load_nd %tdesc_scale_a[0, 0] : !xegpu.tensor_desc<8x2xf8E8M0FNU> -> vector<2xf8E8M0FNU>
      // Option 2: Use load + insert
      %id_x = gpu.thread_id x
      %c8 = arith.constant 8 : index
      %idx_x = arith.remsi %id_x, %c8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %first = memref.load %scale_a[%idx_x, %c0] : memref<8x2xf8E8M0FNU>
      %second = memref.load %scale_a[%idx_x, %c1] : memref<8x2xf8E8M0FNU>
      %scale_a_undef = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %scale_a_tmp = vector.insert %first, %scale_a_undef[%c0] : f8E8M0FNU into vector<2xf8E8M0FNU>
      %scale_a_loaded = vector.insert %second, %scale_a_tmp[%c1] : f8E8M0FNU into vector<2xf8E8M0FNU>

      // scale_b cannot use 2d block load. 8bit load with 16 columns is not supported
      %scale_b_undef = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %first_b = memref.load %scale_b[%c0, %id_x] : memref<2x16xf8E8M0FNU>
      %second_b = memref.load %scale_b[%c1, %id_x] : memref<2x16xf8E8M0FNU>
      %scale_b_tmp = vector.insert %first_b, %scale_b_undef[%c0] : f8E8M0FNU into vector<2xf8E8M0FNU>
      %scale_b_loaded = vector.insert %second_b, %scale_b_tmp[%c1] : f8E8M0FNU into vector<2xf8E8M0FNU>

      %c_result = xegpu.dpas_mx %a_trunc, %b_trunc, %c_loaded scale_a = %scale_a_loaded scale_b = %scale_b_loaded : vector<32xf4E2M1FN>, vector<64xf4E2M1FN>, vector<8xf32>, vector<2xf8E8M0FNU>, vector<2xf8E8M0FNU> -> vector<8xf32>

      xegpu.store_nd %c_result, %tdesc_c[0, 0] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

  func.func @test(%a : memref<8x64xf4E2M1FN>, %b : memref<32x16xi8>, %c : memref<8x16xf32>, %scale_a : memref<8x2xf8E8M0FNU>, %scale_b : memref<2x16xf8E8M0FNU>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<8x64xf4E2M1FN>
    gpu.memcpy %memref_a, %a : memref<8x64xf4E2M1FN>, memref<8x64xf4E2M1FN>

    %memref_b = gpu.alloc() : memref<32x16xi8>
    gpu.memcpy %memref_b, %b : memref<32x16xi8>, memref<32x16xi8>

    %memref_c = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>

    %memref_scale_a = gpu.alloc() : memref<8x2xf8E8M0FNU>
    gpu.memcpy %memref_scale_a, %scale_a : memref<8x2xf8E8M0FNU>, memref<8x2xf8E8M0FNU>

    %memref_scale_b = gpu.alloc() : memref<2x16xf8E8M0FNU>
    gpu.memcpy %memref_scale_b, %scale_b : memref<2x16xf8E8M0FNU>, memref<2x16xf8E8M0FNU>

    gpu.launch_func @kernel::@dpas_mx_e2m1 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
    args(%memref_a : memref<8x64xf4E2M1FN>, %memref_b : memref<32x16xi8>, %memref_c : memref<8x16xf32>, %memref_scale_a : memref<8x2xf8E8M0FNU>, %memref_scale_b : memref<2x16xf8E8M0FNU>)
    gpu.dealloc %memref_a : memref<8x64xf4E2M1FN>
    gpu.dealloc %memref_b : memref<32x16xi8>
    gpu.dealloc %memref_scale_a : memref<8x2xf8E8M0FNU>
    gpu.dealloc %memref_scale_b : memref<2x16xf8E8M0FNU>
    %res = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %res, %memref_c : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_c : memref<8x16xf32>
    return %res : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1e2m1 = arith.constant 1.0 : f4E2M1FN
    %c1packed_e2m1 = arith.constant 0x22 : i8
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    %A = memref.alloc() : memref<8x64xf4E2M1FN>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c1e2m1, %A[%i, %j] : memref<8x64xf4E2M1FN>
      }
    }

    %B = memref.alloc() : memref<32x16xi8>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c1packed_e2m1, %B[%i, %j] : memref<32x16xi8>
      }
    }

    %C = memref.alloc() : memref<8x16xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %scale_A = memref.alloc() : memref<8x2xf8E8M0FNU>
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %c1f8E8M0FNU, %scale_A[%i, %c0] : memref<8x2xf8E8M0FNU>
    }

    %scale_B = memref.alloc() : memref<2x16xf8E8M0FNU>
    scf.for %i = %c0 to %c16 step %c1 {
      memref.store %c1f8E8M0FNU, %scale_B[%c0, %i] : memref<2x16xf8E8M0FNU>
    }

    %C_res = call @test(%A, %B, %C, %scale_A, %scale_B) : (memref<8x64xf4E2M1FN>, memref<32x16xi8>, memref<8x16xf32>, memref<8x2xf8E8M0FNU>, memref<2x16xf8E8M0FNU>) -> memref<8x16xf32>
    %C_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-8: [64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64]
    memref.dealloc %A : memref<8x64xf4E2M1FN>
    memref.dealloc %B : memref<32x16xi8>
    memref.dealloc %C : memref<8x16xf32>
    memref.dealloc %C_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
