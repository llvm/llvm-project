// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// A rank-3 descriptor over a subview of a wider source. The 4 planes of the
// 4x32x32 subview sit 64 rows apart in the 4x64x64 source, so they are not
// packed back to back: the flattened-plane `base_height` the XeGPUToXeVM
// lowering hands to the 2D block store is the row extent 32 + 3 * 64 = 224, and
// every plane has to land inside it for the store to survive the HW boundary
// check.

module @store_nd_3d_subview attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @store_tile(%dst: memref<4x64x64xf32>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<1.000000e+00> : vector<4x32x32xf32>
      %sub = memref.subview %dst[0, 0, 0] [4, 32, 32] [1, 1, 1]
          : memref<4x64x64xf32> to memref<4x32x32xf32, strided<[4096, 64, 1]>>
      %tdesc = xegpu.create_nd_tdesc %sub
          : memref<4x32x32xf32, strided<[4096, 64, 1]>>
          -> !xegpu.tensor_desc<4x32x32xf32>
      xegpu.store_nd %cst, %tdesc[%c0, %c0, %c0]
          <{layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [4, 32, 32]>}>
          : vector<4x32x32xf32>, !xegpu.tensor_desc<4x32x32xf32>
      gpu.return
    }
  }

  func.func @test(%dst: memref<4x64x64xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %dst_gpu = gpu.alloc () : memref<4x64x64xf32>
    gpu.memcpy %dst_gpu, %dst : memref<4x64x64xf32>, memref<4x64x64xf32>
    gpu.launch_func @kernel::@store_tile
        blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%dst_gpu : memref<4x64x64xf32>)
    gpu.wait
    gpu.memcpy %dst, %dst_gpu : memref<4x64x64xf32>, memref<4x64x64xf32>
    gpu.dealloc %dst_gpu : memref<4x64x64xf32>
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %f0 = arith.constant 0.000000e+00 : f32
    // Everything the store does not reach keeps this sentinel.
    %sentinel = arith.constant 9.000000e+00 : f32
    %dst = memref.alloc() : memref<4x64x64xf32>
    scf.for %c = %c0 to %c4 step %c1 {
      scf.for %h = %c0 to %c64 step %c1 {
        scf.for %w = %c0 to %c64 step %c1 {
          memref.store %sentinel, %dst[%c, %h, %w] : memref<4x64x64xf32>
        }
      }
    }
    call @test(%dst) : (memref<4x64x64xf32>) -> ()

    // Sum over the stored tile: 4 * 32 * 32 elements of 1.0. A plane the store
    // did not reach shows up as 32 * 32 * 9.0 too much.
    %tile_sum = scf.for %c = %c0 to %c4 step %c1 iter_args(%acc = %f0) -> f32 {
      %r = scf.for %h = %c0 to %c32 step %c1 iter_args(%acc_h = %acc) -> f32 {
        %r_w = scf.for %w = %c0 to %c32 step %c1 iter_args(%acc_w = %acc_h) -> f32 {
          %v = memref.load %dst[%c, %h, %w] : memref<4x64x64xf32>
          %s = arith.addf %acc_w, %v : f32
          scf.yield %s : f32
        }
        scf.yield %r_w : f32
      }
      scf.yield %r : f32
    }
    // Sum over the whole buffer, so a store that spilled outside the tile is
    // caught too: 4096 * 1.0 + 12288 * 9.0.
    %all_sum = scf.for %c = %c0 to %c4 step %c1 iter_args(%acc = %f0) -> f32 {
      %r = scf.for %h = %c0 to %c64 step %c1 iter_args(%acc_h = %acc) -> f32 {
        %r_w = scf.for %w = %c0 to %c64 step %c1 iter_args(%acc_w = %acc_h) -> f32 {
          %v = memref.load %dst[%c, %h, %w] : memref<4x64x64xf32>
          %s = arith.addf %acc_w, %v : f32
          scf.yield %s : f32
        }
        scf.yield %r_w : f32
      }
      scf.yield %r : f32
    }
    // CHECK: 4096
    vector.print %tile_sum : f32
    // CHECK: 114688
    vector.print %all_sum : f32
    memref.dealloc %dst : memref<4x64x64xf32>
    return
  }
}
