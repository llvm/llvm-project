// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | FileCheck %s

// XFAIL: *

// 3D batched MXFP GEMM: [4, 128, 512] x [4, 512, 128] -> [4, 128, 128]
// with scale_a [4, 128, 16] and scale_b [4, 16, 128]
// Batch dim = 4, split first dim of A to [4, M], last dim of B to [4, N].

#a = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 32, 512], inst_data = [1, 8, 64]>
#b = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 512, 64], inst_data = [1, 64, 16]>
#c = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 32, 64], inst_data = [1, 8, 16]>
// Layouts for the scale operands as consumed by dpas_mx (small inst_data
// sized to the dpas_mx scale tile).
#a_scale = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 32, 16], inst_data = [1, 8, 2]>
#b_scale = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 16, 64], inst_data = [1, 2, 16]>
// Separate layouts for load_nd of the scale tensors: 2D block loads on the
// mx_scale element type require larger inst_data than the dpas_mx operand
// tile, so the load uses its own layout and the values are then re-laid out
// for dpas_mx.
#a_scale_load = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 32, 16], inst_data = [1, 16, 32]>
#b_scale_load = #xegpu.layout<sg_layout = [1, 4, 2], sg_data = [4, 16, 64], inst_data = [1, 32, 16]>

gpu.module @test {
  gpu.func @gemm_3d_mxfp(%arg0: memref<4x128x512xf4E2M1FN>, %arg1: memref<4x512x128xf4E2M1FN>, %arg2: memref<4x128x16xf8E8M0FNU>, %arg3: memref<4x16x128xf8E8M0FNU>, %arg4: memref<4x128x128xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %block_id_z = gpu.block_id z
    %m = arith.muli %block_id_x, %c128 : index
    %n = arith.muli %block_id_y, %c128 : index

    %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<4x128x512xf4E2M1FN> -> !xegpu.tensor_desc<4x128x512xf4E2M1FN>
    %a = xegpu.load_nd %a_tdesc[%block_id_z, %m, %c0] {layout = #a} : !xegpu.tensor_desc<4x128x512xf4E2M1FN> -> vector<4x128x512xf4E2M1FN>

    %b_tdesc = xegpu.create_nd_tdesc %arg1 : memref<4x512x128xf4E2M1FN> -> !xegpu.tensor_desc<4x512x128xf4E2M1FN>
    %b = xegpu.load_nd %b_tdesc[%block_id_z, %c0, %n] {layout = #b} : !xegpu.tensor_desc<4x512x128xf4E2M1FN> -> vector<4x512x128xf4E2M1FN>

    %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<4x128x128xf32> -> !xegpu.tensor_desc<4x128x128xf32, #c>
    %c = xegpu.load_nd %cd_tdesc[%block_id_z, %m, %n] {layout = #c} : !xegpu.tensor_desc<4x128x128xf32, #c> -> vector<4x128x128xf32>

    %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<4x128x16xf8E8M0FNU> -> !xegpu.tensor_desc<4x128x16xf8E8M0FNU>
    %scale_a = xegpu.load_nd %a_scale_tdesc[%block_id_z, %m, %c0] {layout = #a_scale_load} : !xegpu.tensor_desc<4x128x16xf8E8M0FNU> -> vector<4x128x16xf8E8M0FNU>

    %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<4x16x128xf8E8M0FNU> -> !xegpu.tensor_desc<4x16x128xf8E8M0FNU>
    %scale_b = xegpu.load_nd %b_scale_tdesc[%block_id_z, %c0, %n] {layout = #b_scale_load} : !xegpu.tensor_desc<4x16x128xf8E8M0FNU> -> vector<4x16x128xf8E8M0FNU>

    %d = xegpu.dpas_mx %a, %b, %c scale_a = %scale_a scale_b = %scale_b
          {layout_a = #a,
           layout_b = #b,
           layout_cd = #c,
           layout_a_scale = #a_scale,
           layout_b_scale = #b_scale}
        : (vector<4x128x512xf4E2M1FN>, vector<4x512x128xf4E2M1FN>,
          vector<4x128x128xf32>,
          vector<4x128x16xf8E8M0FNU>, vector<4x16x128xf8E8M0FNU>)
        -> vector<4x128x128xf32>

    xegpu.store_nd %d, %cd_tdesc[%block_id_z, %m, %n] {layout = #c} : vector<4x128x128xf32>, !xegpu.tensor_desc<4x128x128xf32, #c>
    gpu.return
  }
}
