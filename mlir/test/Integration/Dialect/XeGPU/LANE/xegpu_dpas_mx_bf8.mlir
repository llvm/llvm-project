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
    gpu.func @dpas_mx_bf8(%a: memref<8x32xf16>, %b: memref<32x16xf8E5M2>, %c: memref<8x16xf32>, %scale_a: memref<8x1xf8E8M0FNU>, %scale_b: memref<1x16xf8E8M0FNU>) kernel {

      %tdesc_a = xegpu.create_nd_tdesc %a : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      %loaded_a = xegpu.load_nd %tdesc_a[0, 0] : !xegpu.tensor_desc<8x32xf16> -> vector<16xf16>
      %a_trunc = xegpu.truncf %loaded_a : vector<16xf16> -> vector<16xf8E5M2>

      %tdesc_b = xegpu.create_nd_tdesc %b : memref<32x16xf8E5M2> -> !xegpu.tensor_desc<32x16xf8E5M2>
      %b_trunc = xegpu.load_nd %tdesc_b[0, 0] : !xegpu.tensor_desc<32x16xf8E5M2> -> vector<32xf8E5M2>

      %tdesc_c = xegpu.create_nd_tdesc %c : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %c_loaded = xegpu.load_nd %tdesc_c[0, 0] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

      %tdesc_scale_a = xegpu.create_nd_tdesc %scale_a : memref<8x1xf8E8M0FNU> -> !xegpu.tensor_desc<8x1xf8E8M0FNU>
      %scale_a_loaded = xegpu.load_nd %tdesc_scale_a[0, 0] : !xegpu.tensor_desc<8x1xf8E8M0FNU> -> vector<1xf8E8M0FNU>

      %tdesc_scale_b = xegpu.create_nd_tdesc %scale_b : memref<1x16xf8E8M0FNU> -> !xegpu.tensor_desc<1x16xf8E8M0FNU>
      %scale_b_loaded = xegpu.load_nd %tdesc_scale_b[0, 0] : !xegpu.tensor_desc<1x16xf8E8M0FNU> -> vector<1xf8E8M0FNU>

      %c_result = xegpu.dpas_mx %a_trunc, %b_trunc, %c_loaded scale_a = %scale_a_loaded scale_b = %scale_b_loaded : vector<16xf8E5M2>, vector<32xf8E5M2>, vector<8xf32>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> -> vector<8xf32>

      xegpu.store_nd %c_result, %tdesc_c[0, 0] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

  func.func @test(%a : memref<8x32xf16>, %b : memref<32x16xf8E5M2>, %c : memref<8x16xf32>, %scale_a : memref<8x1xf8E8M0FNU>, %scale_b : memref<1x16xf8E8M0FNU>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<8x32xf16>
    gpu.memcpy %memref_a, %a : memref<8x32xf16>, memref<8x32xf16>

    %memref_b = gpu.alloc() : memref<32x16xf8E5M2>
    gpu.memcpy %memref_b, %b : memref<32x16xf8E5M2>, memref<32x16xf8E5M2>

    %memref_c = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>

    %memref_scale_a = gpu.alloc() : memref<8x1xf8E8M0FNU>
    gpu.memcpy %memref_scale_a, %scale_a : memref<8x1xf8E8M0FNU>, memref<8x1xf8E8M0FNU>

    %memref_scale_b = gpu.alloc() : memref<1x16xf8E8M0FNU>
    gpu.memcpy %memref_scale_b, %scale_b : memref<1x16xf8E8M0FNU>, memref<1x16xf8E8M0FNU>

    gpu.launch_func @kernel::@dpas_mx_bf8 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
    args(%memref_a : memref<8x32xf16>, %memref_b : memref<32x16xf8E5M2>, %memref_c : memref<8x16xf32>, %memref_scale_a : memref<8x1xf8E8M0FNU>, %memref_scale_b : memref<1x16xf8E8M0FNU>)
    gpu.dealloc %memref_a : memref<8x32xf16>
    gpu.dealloc %memref_b : memref<32x16xf8E5M2>
    gpu.dealloc %memref_scale_a : memref<8x1xf8E8M0FNU>
    gpu.dealloc %memref_scale_b : memref<1x16xf8E8M0FNU>
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
    %c1f16 = arith.constant 1.0 : f16
    %c1bf8 = arith.constant 1.0 : f8E5M2
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    %A = memref.alloc() : memref<8x32xf16>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c1f16, %A[%i, %j] : memref<8x32xf16>
      }
    }

    %B = memref.alloc() : memref<32x16xf8E5M2>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c1bf8, %B[%i, %j] : memref<32x16xf8E5M2>
      }
    }

    %C = memref.alloc() : memref<8x16xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %scale_A = memref.alloc() : memref<8x1xf8E8M0FNU>
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %c1f8E8M0FNU, %scale_A[%i, %c0] : memref<8x1xf8E8M0FNU>
    }

    %scale_B = memref.alloc() : memref<1x16xf8E8M0FNU>
    scf.for %i = %c0 to %c16 step %c1 {
      memref.store %c1f8E8M0FNU, %scale_B[%c0, %i] : memref<1x16xf8E8M0FNU>
    }

    %C_res = call @test(%A, %B, %C, %scale_A, %scale_B) : (memref<8x32xf16>, memref<32x16xf8E5M2>, memref<8x16xf32>, memref<8x1xf8E8M0FNU>, memref<1x16xf8E8M0FNU>) -> memref<8x16xf32>
    %C_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-8: [32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   32]
    memref.dealloc %A : memref<8x32xf16>
    memref.dealloc %B : memref<32x16xf8E5M2>
    memref.dealloc %C : memref<8x16xf32>
    memref.dealloc %C_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
