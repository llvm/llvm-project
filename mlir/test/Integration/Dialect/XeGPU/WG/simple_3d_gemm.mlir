// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | FileCheck %s

// XFAIL: *

#a = #xegpu.layout<sg_layout = [1, 8, 4], sg_data = [4, 8, 32], inst_data = [1, 8, 16]>
#b = #xegpu.layout<sg_layout = [1, 8, 4], sg_data = [4, 32, 16], inst_data = [1, 16, 16]>
#c = #xegpu.layout<sg_layout = [1, 8, 4], sg_data = [4, 8, 16], inst_data = [1, 8, 16]>
#a_prefetch = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 8, 32], inst_data = [1, 8, 16]>
#b_prefetch = #xegpu.layout<sg_layout = [4, 4, 2], sg_data = [1, 8, 32], inst_data = [1, 8, 16]>

gpu.module @test_kernel {
  gpu.func @test_kernel(%A: memref<4x64x256xf16>, %B: memref<4x256x64xf16>, %C: memref<4x64x64xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %c256 = arith.constant 256 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %block_id_z = gpu.block_id z
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c64 : index
    %c_tdesc = xegpu.create_nd_tdesc %C : memref<4x64x64xf32> -> !xegpu.tensor_desc<4x64x64xf32, #c>
    %c_init_value = xegpu.load_nd %c_tdesc[%block_id_z, %m, %n] {layout = #c} : !xegpu.tensor_desc<4x64x64xf32, #c> -> vector<4x64x64xf32>
    %a_tdesc = xegpu.create_nd_tdesc %A : memref<4x64x256xf16> -> !xegpu.tensor_desc<4x64x32xf16, #a>
    %b_tdesc = xegpu.create_nd_tdesc %B : memref<4x256x64xf16> -> !xegpu.tensor_desc<4x32x64xf16, #b>
    // Prefetch A 3 times.
    %a_prefetch_tdesc = xegpu.create_nd_tdesc %A : memref<4x64x256xf16> -> !xegpu.tensor_desc<4x64x32xf16, #a_prefetch>
    xegpu.prefetch_nd %a_prefetch_tdesc[%block_id_z, %m, %c0] {layout = #a_prefetch} : !xegpu.tensor_desc<4x64x32xf16, #a_prefetch>
    xegpu.prefetch_nd %a_prefetch_tdesc[%block_id_z, %m, %c32] {layout = #a_prefetch} : !xegpu.tensor_desc<4x64x32xf16, #a_prefetch>
    xegpu.prefetch_nd %a_prefetch_tdesc[%block_id_z, %m, %c64] {layout = #a_prefetch} : !xegpu.tensor_desc<4x64x32xf16, #a_prefetch>
    // Prefetch B 3 times.
    %b_prefetch_tdesc = xegpu.create_nd_tdesc %B : memref<4x256x64xf16> -> !xegpu.tensor_desc<4x32x64xf16, #b_prefetch>
    xegpu.prefetch_nd %b_prefetch_tdesc[%block_id_z, %c0, %n] {layout = #b_prefetch} : !xegpu.tensor_desc<4x32x64xf16, #b_prefetch>
    xegpu.prefetch_nd %b_prefetch_tdesc[%block_id_z, %c32, %n] {layout = #b_prefetch} : !xegpu.tensor_desc<4x32x64xf16, #b_prefetch>
    xegpu.prefetch_nd %b_prefetch_tdesc[%block_id_z, %c64, %n] {layout = #b_prefetch} : !xegpu.tensor_desc<4x32x64xf16, #b_prefetch>

    %out = scf.for %k = %c0 to %c256 step %c32
      iter_args(%c_value = %c_init_value)
      -> (vector<4x64x64xf32>) {
      %a_value = xegpu.load_nd %a_tdesc[%block_id_z, %m, %k] {layout = #a} : !xegpu.tensor_desc<4x64x32xf16, #a> -> vector<4x64x32xf16>
      %b_value = xegpu.load_nd %b_tdesc[%block_id_z, %k, %n] {layout = #b} : !xegpu.tensor_desc<4x32x64xf16, #b> -> vector<4x32x64xf16>
      // Prefetch next tiles.
      %prefetch_offset = arith.addi %k, %c96 : index
      xegpu.prefetch_nd %a_prefetch_tdesc[%block_id_z, %m, %prefetch_offset] {layout = #a_prefetch} : !xegpu.tensor_desc<4x64x32xf16, #a_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%block_id_z, %prefetch_offset, %n] {layout = #b_prefetch} : !xegpu.tensor_desc<4x32x64xf16, #b_prefetch>
      %c_new_value = xegpu.dpas %a_value, %b_value, %c_value {layout_a = #a, layout_b = #b, layout_cd = #c}
        : vector<4x64x32xf16>, vector<4x32x64xf16>, vector<4x64x64xf32> -> vector<4x64x64xf32>
      scf.yield %c_new_value : vector<4x64x64xf32>
    }
    xegpu.store_nd %out, %c_tdesc[%block_id_z, %m, %n] {layout = #c} : vector<4x64x64xf32>, !xegpu.tensor_desc<4x64x64xf32, #c>
    gpu.return
  }
}
