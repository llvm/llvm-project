// RUN: mlir-opt -xevm-attach-target='chip=pvc' -xegpu-propagate-layout="layout-kind=inst" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @dpas_f16(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} dense<0.000000e+00> : vector<8x16xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<inst_data = [8, 16]>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]][{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]]  {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} :
// CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T1]]  {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>> -> vector<16x16xf16>
// CHECK: %[[T4:.*]] = xegpu.dpas %[[T2]], %[[T3]], %[[CST]] {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} :
// CHECK-SAME: vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<inst_data = [8, 16]>
// CHECK: xegpu.store_nd %[[T4]], %[[T5]] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<inst_data = [8, 16]>>
gpu.module @test {

func.func @dpas_f16(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}
}

// -----
gpu.module @test_kernel {
  gpu.func @elementwise_with_inst_data_only(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>
    %b_tdesc = xegpu.create_nd_tdesc %B[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>
    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>

    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_tdesc)
      -> (!xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>) {
      //CHECK: xegpu.load_nd {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} :
      //CHECK-SAME: !xegpu.tensor_desc<16x32xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x32xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<16x32xf16> -> vector<16x32xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<16x32xf16> -> vector<16x32xf16>

      //CHECK-COUNT: arith.addf {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x32xf16>
      %c = arith.addf %a, %b : vector<16x32xf16>

      //CHECK-COUNT: xegpu.store_nd {{.*}} : vector<16x32xf16>, !xegpu.tensor_desc<16x32xf16, #xegpu.layout<inst_data = [8, 16]>>
      xegpu.store_nd %c, %arg2: vector<16x32xf16>, !xegpu.tensor_desc<16x32xf16>

      //CHECK-COUNT: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16x32xf16, #xegpu.layout<inst_data = [8, 16]>>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16>
      %c_next_tdesc = xegpu.update_nd_offset %arg2, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c_next_tdesc
        : !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>, !xegpu.tensor_desc<16x32xf16>
    }
    gpu.return
  }
}

// -----
gpu.module @test_kernel {
  gpu.func @elementwise_with_inst_data_12(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>
    %b_tdesc = xegpu.create_nd_tdesc %B[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>
    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>

    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_tdesc)
      -> (!xegpu.tensor_desc<12x32xf16>, !xegpu.tensor_desc<12x32xf16>, !xegpu.tensor_desc<12x32xf16>) {
      //CHECK: xegpu.load_nd {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>} :
      //CHECK-SAME: !xegpu.tensor_desc<12x32xf16, #xegpu.layout<inst_data = [4, 16]>> -> vector<12x32xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<12x32xf16> -> vector<12x32xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<12x32xf16> -> vector<12x32xf16>

      //CHECK-COUNT: arith.addf {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>} : vector<12x32xf16>
      %c = arith.addf %a, %b : vector<12x32xf16>

      //CHECK-COUNT: xegpu.store_nd {{.*}} : vector<12x32xf16>, !xegpu.tensor_desc<12x32xf16, #xegpu.layout<inst_data = [4, 16]>>
      xegpu.store_nd %c, %arg2: vector<12x32xf16>, !xegpu.tensor_desc<12x32xf16>

      //CHECK-COUNT: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<12x32xf16, #xegpu.layout<inst_data = [4, 16]>>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<12x32xf16>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c0, %c32] : !xegpu.tensor_desc<12x32xf16>
      %c_next_tdesc = xegpu.update_nd_offset %arg2, [%c0, %c32] : !xegpu.tensor_desc<12x32xf16>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c_next_tdesc
        : !xegpu.tensor_desc<12x32xf16>, !xegpu.tensor_desc<12x32xf16>, !xegpu.tensor_desc<12x32xf16>
    }
    gpu.return
  }
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @scatter_ops_chunksize(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf16>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<true> : vector<16xi1>
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<12> : vector<16xindex>
// CHECK: %{{.*}} = xegpu.load %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 8 : i64}>
// CHECK-SAME: {layout_result_0 = #xegpu.layout<inst_data = [16, 8]>} : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
// CHECK: xegpu.store %0, %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 8 : i64}> : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
func.func @scatter_ops_chunksize(%src: memref<256xf16>) {
  %1 = arith.constant dense<1>: vector<16xi1>
  %offset = arith.constant dense<12> : vector<16xindex>
  %3 = xegpu.load %src[%offset], %1 <{chunk_size=8}>
      : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
  xegpu.store %3, %src[%offset], %1 <{chunk_size=8}>
      : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
  return
}
}
