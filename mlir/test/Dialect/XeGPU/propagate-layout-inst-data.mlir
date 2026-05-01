// RUN: mlir-opt -xevm-attach-target='chip=cri' -test-xegpu-propagate-layouts="layout-kind=inst" -split-input-file %s | FileCheck %s


// CHECK-LABEL: func.func @load_store_no_array_len(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x32xf32>, %[[ARG1:[0-9a-zA-Z]+]]: memref<8x32xf32>) {
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK: %[[TDESC_SRC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: %[[TDESC_DST:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: xegpu.prefetch_nd %[[TDESC_SRC]][0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: %[[LOADED:.*]] = xegpu.load_nd %0[0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}>
// CHECK-SAME: !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>> -> vector<8x32xf32>
// CHECK: xegpu.store_nd %[[LOADED]], %[[TDESC_DST]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
gpu.module @test {
// Although the uArch allows 8x32 inst data using block count (or array_len),
// it is up to optimization passes to decide on the block count usage.
func.func @load_store_no_array_len(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32>
  xegpu.prefetch_nd %0[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<8x32xf32>
  %2 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<8x32xf32> -> vector<8x32xf32>
  xegpu.store_nd %2, %1[0, 0]  : vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32>
  return
}
}

// -----

// CHECK-LABEL: func.func @dpas_f16(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} dense<0.000000e+00> : vector<8x16xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<inst_data = [8, 16]>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]][0, 0]  <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T1]][0, 0] <{layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>> -> vector<16x16xf16>
// CHECK: %[[T4:.*]] = xegpu.dpas %[[T2]], %[[T3]], %[[CST]] {layout_a = #xegpu.layout<inst_data = [8, 16]>, layout_b = #xegpu.layout<inst_data = [16, 16]>, layout_cd = #xegpu.layout<inst_data = [8, 16]>} :
// CHECK-SAME: vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<inst_data = [8, 16]>
// CHECK: xegpu.store_nd %[[T4]], %[[T5]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<inst_data = [8, 16]>>
gpu.module @test {
func.func @dpas_f16(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1[0, 0]  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5[0, 0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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

    %a_tdesc = xegpu.create_nd_tdesc %A : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>
    %b_tdesc = xegpu.create_nd_tdesc %B : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>
    %c_tdesc = xegpu.create_nd_tdesc %C : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16>

    scf.for %k = %c0 to %c1024 step %c32 {
      //CHECK: xegpu.load_nd {{.*}} <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
      //CHECK-SAME: !xegpu.tensor_desc<16x32xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x32xf16>
      %a = xegpu.load_nd %a_tdesc[0, %k] : !xegpu.tensor_desc<16x32xf16> -> vector<16x32xf16>
      %b = xegpu.load_nd %b_tdesc[0, %k] : !xegpu.tensor_desc<16x32xf16> -> vector<16x32xf16>

      //CHECK-COUNT: arith.addf {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<16x32xf16>
      %c = arith.addf %a, %b : vector<16x32xf16>

      //CHECK-COUNT: xegpu.store_nd {{.*}} : vector<16x32xf16>, !xegpu.tensor_desc<16x32xf16, #xegpu.layout<inst_data = [8, 16]>>
      xegpu.store_nd %c, %c_tdesc[0, %k] : vector<16x32xf16>, !xegpu.tensor_desc<16x32xf16>
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

    %a_tdesc = xegpu.create_nd_tdesc %A : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>
    %b_tdesc = xegpu.create_nd_tdesc %B : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>
    %c_tdesc = xegpu.create_nd_tdesc %C : memref<1024x1024xf16> -> !xegpu.tensor_desc<12x32xf16>

    scf.for %k = %c0 to %c1024 step %c32 {
      //CHECK: xegpu.load_nd {{.*}} <{layout = #xegpu.layout<inst_data = [4, 16]>}> :
      //CHECK-SAME: !xegpu.tensor_desc<12x32xf16, #xegpu.layout<inst_data = [4, 16]>> -> vector<12x32xf16>
      %a = xegpu.load_nd %a_tdesc[0, %k] : !xegpu.tensor_desc<12x32xf16> -> vector<12x32xf16>
      %b = xegpu.load_nd %b_tdesc[0, %k] : !xegpu.tensor_desc<12x32xf16> -> vector<12x32xf16>

      //CHECK-COUNT: arith.addf {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>} : vector<12x32xf16>
      %c = arith.addf %a, %b : vector<12x32xf16>

      //CHECK-COUNT: xegpu.store_nd {{.*}} : vector<12x32xf16>, !xegpu.tensor_desc<12x32xf16, #xegpu.layout<inst_data = [4, 16]>>
      xegpu.store_nd %c, %c_tdesc[0, %k] : vector<12x32xf16>, !xegpu.tensor_desc<12x32xf16>
    }
    gpu.return
  }
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @scatter_ops_chunksize(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf16>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<true> : vector<16xi1>
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<12> : vector<16xindex>
// CHECK: %{{.*}} = xegpu.load %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 8 : i64, layout = #xegpu.layout<inst_data = [16, 8]>}>
// CHECK-SAME: memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
// CHECK: xegpu.store %0, %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 8 : i64, layout = #xegpu.layout<inst_data = [16, 8]>}> : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
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

// -----
gpu.module @test {
// CHECK-LABEL: func.func @store_matrix(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.mem_desc<16x64xf16>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [1, 16]>} dense<0.000000e+00> : vector<16x16xf16>
func.func @store_matrix(%arg0: !xegpu.mem_desc<16x64xf16>) {
  %cst = arith.constant dense<0.0000> : vector<16x16xf16>
  xegpu.store_matrix %cst, %arg0[8, 8]: vector<16x16xf16>, !xegpu.mem_desc<16x64xf16>

  return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @scatter_ops_chunksize_excessive(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<1024xf32>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<true> : vector<16xi1>
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<12> : vector<16xindex>
// CHECK: %{{.*}} = xegpu.load %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 32 : i64, layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16x32xf32>
// CHECK: xegpu.store %0, %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 32 : i64, layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: vector<16x32xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
func.func @scatter_ops_chunksize_excessive(%src: memref<1024xf32>) {
  %1 = arith.constant dense<1>: vector<16xi1>
  %offset = arith.constant dense<12> : vector<16xindex>
  %3 = xegpu.load %src[%offset], %1 <{chunk_size=32}>
      : memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16x32xf32>
  xegpu.store %3, %src[%offset], %1 <{chunk_size=32}>
      : vector<16x32xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
  return
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @scatter_ops_chunksize_excessive_anchor(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<1024xf32>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<true> : vector<16xi1>
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<12> : vector<16xindex>
// CHECK: %{{.*}} = xegpu.load %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 32 : i64, layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16x32xf32>
// CHECK: xegpu.store %0, %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 32 : i64, layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: vector<16x32xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
func.func @scatter_ops_chunksize_excessive_anchor(%src: memref<1024xf32>) {
  %1 = arith.constant dense<1>: vector<16xi1>
  %offset = arith.constant dense<12> : vector<16xindex>
  %3 = xegpu.load %src[%offset], %1 <{chunk_size=32}>
      : memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16x32xf32>
  xegpu.store %3, %src[%offset], %1 <{chunk_size=32, layout = #xegpu.layout<inst_data = [16, 16]>}>
      : vector<16x32xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
  return
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @scatter_ops_chunksize_slice(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<1024xf32>) {
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<true> : vector<16xi1>
// CHECK: %{{.*}} = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [16]>} dense<12> : vector<16xindex>
// CHECK: %[[LOADED:.*]] = xegpu.load %[[ARG0]][%{{.*}}], %{{.*}} <{layout = #xegpu.layout<inst_data = [16]>}> :
// CHECK-SAME: memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16xf32>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[LOADED]] {layout_result_0 = #xegpu.layout<inst_data = [16, 16]>} : vector<16xf32> to vector<16x16xf32>
// CHECK: xegpu.store %[[BCAST]], %[[ARG0]][%{{.*}}], %{{.*}} <{chunk_size = 16 : i64, layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME: vector<16x16xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
func.func @scatter_ops_chunksize_slice(%src: memref<1024xf32>) {
  %1 = arith.constant dense<1>: vector<16xi1>
  %offset = arith.constant dense<12> : vector<16xindex>
  %3 = xegpu.load %src[%offset], %1
      : memref<1024xf32>, vector<16xindex>, vector<16xi1> -> vector<16xf32>

  %4 = vector.broadcast %3 : vector<16xf32> to vector<16x16xf32>
  xegpu.store %4, %src[%offset], %1 <{chunk_size=16, layout = #xegpu.layout<inst_data = [16, 16]>}>
      : vector<16x16xf32>, memref<1024xf32>, vector<16xindex>, vector<16xi1>
  return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @insert_strided_slice_inst_data_no_packing(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x32xf32>) {
// CHECK: %[[CST_SMALL:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>} dense<1.000000e+00> : vector<4x16xf32>
// CHECK: %[[CST_LARGE:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>} dense<0.000000e+00> : vector<8x32xf32>
// CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[CST_SMALL]], %[[CST_LARGE]] {layout_result_0 = #xegpu.layout<inst_data = [4, 16]>, offsets = [0, 0], strides = [1, 1]} : vector<4x16xf32> into vector<8x32xf32>
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: xegpu.store_nd %[[INSERT]], %[[TDESC]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32, #xegpu.layout<inst_data = [8, 16]>>
func.func @insert_strided_slice_inst_data_no_packing(%arg0: memref<8x32xf32>) {
  %c0 = arith.constant 0 : index
  %cst_small = arith.constant dense<1.0> : vector<4x16xf32>
  %cst_large = arith.constant dense<0.0> : vector<8x32xf32>
  %insert = vector.insert_strided_slice %cst_small, %cst_large {offsets = [0, 0], strides = [1, 1]} : vector<4x16xf32> into vector<8x32xf32>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<8x32xf32> -> !xegpu.tensor_desc<8x32xf32>
  xegpu.store_nd %insert, %tdesc[0, 0] : vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32>
  return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @insert_strided_slice_inst_data_with_packing(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x64xi8>) {
// CHECK: %[[CST_SMALL:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [4, 64]>} dense<1> : vector<4x64xi8>
// CHECK: %[[CST_LARGE:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [4, 64]>} dense<0> : vector<8x64xi8>
// CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[CST_SMALL]], %[[CST_LARGE]] {layout_result_0 = #xegpu.layout<inst_data = [4, 64]>, offsets = [0, 0], strides = [1, 1]} : vector<4x64xi8> into vector<8x64xi8>
func.func @insert_strided_slice_inst_data_with_packing(%arg0: memref<8x64xi8>) {
  %c0 = arith.constant 0 : index
  %cst_small = arith.constant dense<1> : vector<4x64xi8>
  %cst_large = arith.constant dense<0> : vector<8x64xi8>
  %insert = vector.insert_strided_slice %cst_small, %cst_large {offsets = [0, 0], strides = [1, 1]} : vector<4x64xi8> into vector<8x64xi8>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<8x64xi8> -> !xegpu.tensor_desc<8x64xi8, #xegpu.layout<inst_data = [8, 64]>>
  xegpu.store_nd %insert, %tdesc[0, 0] <{layout = #xegpu.layout<inst_data = [8, 64]>}>: vector<8x64xi8>, !xegpu.tensor_desc<8x64xi8, #xegpu.layout<inst_data = [8, 64]>>
  return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @vector_shape_cast_expand_non_unit_dims(
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[STEP:.*]]], %[[CST:.*]] <{layout = #xegpu.layout<inst_data = [16]>}> : memref<1024xf16>, vector<1024xindex>, vector<1024xi1> -> vector<1024xf16>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 16]>} : vector<1024xf16> to vector<8x8x16xf16>
// CHECK: %[[CST_0:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16]>, dims = [0]>} dense<0.000000e+00> : vector<8x16xf16>
// CHECK: %[[CST_1:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 16]>, dims = [0]>} dense<0.000000e+00> : vector<16xf16>
// CHECK: %[[REDUCE_0:.*]] = vector.multi_reduction <add>, %[[CAST]], %[[CST_0]] {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16]>, dims = [0]>} [0] : vector<8x8x16xf16> to vector<8x16xf16>
// CHECK: %[[REDUCE_1:.*]] = vector.multi_reduction <add>, %[[REDUCE_0]], %[[CST_1]] {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 16]>, dims = [0]>} [0] : vector<8x16xf16> to vector<16xf16>
func.func @vector_shape_cast_expand_non_unit_dims(%arg0: memref<1024xf16>, %arg1: memref<16xf16>) {
    %cst = arith.constant dense<true> : vector<1024xi1>
    %0 = vector.step : vector<1024xindex>
    %1 = xegpu.load %arg0[%0], %cst  : memref<1024xf16>, vector<1024xindex>, vector<1024xi1> -> vector<1024xf16>
    %2 = vector.shape_cast %1 : vector<1024xf16> to vector<8x8x16xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8x16xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf16>
    %3 = vector.multi_reduction <add>, %2, %cst_0 [0] : vector<8x8x16xf16> to vector<8x16xf16>
    %4 = vector.multi_reduction <add>, %3, %cst_1 [0] : vector<8x16xf16> to vector<16xf16>
    %cst_2 = arith.constant dense<true> : vector<16xi1>
    %cst_3 = arith.constant dense<1> : vector<16xindex>
    xegpu.store %4, %arg1[%cst_3], %cst_2 <{layout = #xegpu.layout<inst_data = [16]>}> : vector<16xf16>, memref<16xf16>, vector<16xindex>, vector<16xi1>
    return
  }
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @vector_2d_reduction_with_fractional_subgroup_size(
// CHECK: %[[ReduceVal:.*]] = vector.multi_reduction <add>, %[[Val:.*]], %[[CST:.*]] {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 1]>, dims = [1, 2]>} [1, 2] : vector<1x16x1xf16> to vector<1xf16>
func.func @vector_2d_reduction_with_fractional_subgroup_size(%arg0: memref<1024xf16>, %arg1: memref<16xf16>) {
    %cst = arith.constant dense<true> : vector<16xi1>
    %0 = vector.step : vector<16xindex>
    %1 = xegpu.load %arg0[%0], %cst  : memref<1024xf16>, vector<16xindex>, vector<16xi1> -> vector<16xf16>
    %2 = vector.shape_cast %1 : vector<16xf16> to vector<1x16x1xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
    %4 = vector.multi_reduction <add>, %2, %cst_0 [1, 2] : vector<1x16x1xf16> to vector<1xf16>
    %cst_2 = arith.constant dense<true> : vector<1xi1>
    %cst_3 = arith.constant dense<1> : vector<1xindex>
    xegpu.store %4, %arg1[%cst_3], %cst_2 <{layout = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16]>, dims = [1, 2]>}> : vector<1xf16>, memref<16xf16>, vector<1xindex>, vector<1xi1>
    return
  }
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @vector_2d_reduction_with_fractional_subgroup_size_1x4x1(
// CHECK: %[[ReduceVal:.*]] = vector.multi_reduction <add>, %[[Val:.*]], %[[CST:.*]] {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 4]>, dims = [1, 2]>} [1, 2] : vector<1x16x4xf16> to vector<1xf16>
func.func @vector_2d_reduction_with_fractional_subgroup_size_1x4x1(%arg0: memref<1024xf16>, %arg1: memref<16xf16>) {
    %cst = arith.constant dense<true> : vector<64xi1>
    %0 = vector.step : vector<64xindex>
    %1 = xegpu.load %arg0[%0], %cst  : memref<1024xf16>, vector<64xindex>, vector<64xi1> -> vector<64xf16>
    %2 = vector.shape_cast %1 : vector<64xf16> to vector<1x16x4xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
    %4 = vector.multi_reduction <add>, %2, %cst_0 [1, 2] : vector<1x16x4xf16> to vector<1xf16>
    %cst_2 = arith.constant dense<true> : vector<1xi1>
    %cst_3 = arith.constant dense<1> : vector<1xindex>
    xegpu.store %4, %arg1[%cst_3], %cst_2 <{layout = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16]>, dims = [1, 2]>}> : vector<1xf16>, memref<16xf16>, vector<1xindex>, vector<1xi1>
    return
  }
}

// -----
gpu.module @test {
// CHECK-LABEL: func.func @vector_shape_cast_expand_and_merge(
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [32]>} dense<true> : vector<256xi1>
// CHECK: %[[STEP:.*]] = vector.step {layout_result_0 = #xegpu.layout<inst_data = [32]>} : vector<256xindex>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[STEP]]], %[[CST]] <{layout = #xegpu.layout<inst_data = [32]>}> : memref<256xf16>, vector<256xindex>, vector<256xi1> -> vector<256xf16>
// CHECK: %[[CAST_0:.*]] = vector.shape_cast %[[LOAD]] {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 32]>} : vector<256xf16> to vector<2x4x32xf16>
// CHECK: %[[CAST_1:.*]] = vector.shape_cast %[[CAST_0]] {layout_result_0 = #xegpu.layout<inst_data = [1, 32]>} : vector<2x4x32xf16> to vector<1x256xf16>
// CHECK: %[[CAST_2:.*]] = vector.shape_cast %[[CAST_1]] {layout_result_0 = #xegpu.layout<inst_data = [32]>} : vector<1x256xf16> to vector<256xf16>
// CHECK: xegpu.store %[[CAST_2]], %arg1[%[[STEP]]], %[[CST]] <{layout = #xegpu.layout<inst_data = [32]>}> : vector<256xf16>, memref<256xf16>, vector<256xindex>, vector<256xi1>
func.func @vector_shape_cast_expand_and_merge(%arg0: memref<256xf16>, %arg1: memref<256xf16>) {
    %cst = arith.constant dense<true> : vector<256xi1>
    %0 = vector.step : vector<256xindex>
    %1 = xegpu.load %arg0[%0], %cst : memref<256xf16>, vector<256xindex>, vector<256xi1> -> vector<256xf16>
    %2 = vector.shape_cast %1 : vector<256xf16> to vector<2x4x32xf16>

    %4 = vector.shape_cast %2 : vector<2x4x32xf16> to vector<1x256xf16>
    %5 = vector.shape_cast %4 : vector<1x256xf16> to vector<256xf16>
    xegpu.store %5, %arg1[%0], %cst <{layout = #xegpu.layout<inst_data = [32] >}> : vector<256xf16>, memref<256xf16>, vector<256xindex>, vector<256xi1>
    return
  }
}

// -----
gpu.module @test{
  // CHECK-LABEL: load_store_matrix
  // CHECK: xegpu.load_matrix %{{.*}} <{layout = #xegpu.layout<inst_data = [1, 1]>}>
  // CHECK: xegpu.store_matrix %{{.*}} <{layout = #xegpu.layout<inst_data = [1, 1]>}>
  func.func @load_store_matrix(%arg0: !xegpu.mem_desc<64x128xf32>, %arg1: i1) {
    %c0 = arith.constant 0 : index
    scf.if %arg1 {
      %0 = xegpu.load_matrix %arg0[%c0, %c0] : !xegpu.mem_desc<64x128xf32>, index, index -> vector<2x1xf32>
      xegpu.store_matrix %0, %arg0[%c0, %c0] : vector<2x1xf32>, !xegpu.mem_desc<64x128xf32>, index, index
    }
    return
  }
}

// -----
gpu.module @test{
  // CHECK-LABEL: broadcast_both_leadingdims_innerdims
  // CHECK: arith.constant {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 1, 16]>} dense<true> : vector<2x2x6x32xi1>
  // CHECK: arith.constant {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 1, 16]>} dense<1.000000e+00> : vector<2x2x6x32xf32>
  // CHECK: vector.step {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<inst_data = [1, 1, 1, 1]>, dims = [0, 1]>, dims = [1]>} : vector<6xindex>
  // CHECK: vector.shape_cast {{.*}} {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 1, 1]>, dims = [0, 1]>} : vector<6xindex> to vector<6x1xindex>
  // CHECK: vector.broadcast {{.*}} {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 1, 16]>} : vector<6x1xindex> to vector<2x2x6x32xindex>
  gpu.func @broadcast_both_leadingdims_innerdims(%arg0: memref<32x2x192xf32>, %arg1: memref<32x2x192xf32>, %arg2: memref<32x2x192xf32>) kernel attributes {known_block_size = array<i32: 768, 1, 1>, known_grid_size = array<i32: 16, 1, 1>} {
    %cst = arith.constant dense<true> : vector<2x2x6x32xi1>
    %cst_0 = arith.constant dense<1.000000e+00> : vector<2x2x6x32xf32>
    %intptr = memref.extract_aligned_pointer_as_index %arg2 : memref<32x2x192xf32> -> index
    %0 = arith.index_cast %intptr : index to i64
    %1 = vector.step : vector<6xindex>
    %2 = vector.shape_cast %1 : vector<6xindex> to vector<6x1xindex>
    %3 = vector.broadcast %2 : vector<6x1xindex> to vector<2x2x6x32xindex>
    xegpu.store %cst_0, %0[%3], %cst <{layout = #xegpu.layout<inst_data = [1, 1, 1, 16]>}> : vector<2x2x6x32xf32>, i64, vector<2x2x6x32xindex>, vector<2x2x6x32xi1>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @shape_cast_collapse_dims_with_order
gpu.module @test_collapse_dims [#xevm.target<O = 3, chip = "pvc">] {
  gpu.func @shape_cast_collapse_dims_with_order(%arg0: memref<32x32xf32>) {
    // CHECK: %[[STEP:.*]] = vector.step {{.*}} : vector<1024xindex>
    %0 = vector.step : vector<1024xindex>

    // Shape cast from 1D to 2D triggers collapseDims in layout propagation
    // CHECK: %[[CAST:.*]] = vector.shape_cast %[[STEP]] {{.*}} : vector<1024xindex> to vector<32x32xindex>
    %1 = vector.shape_cast %0 : vector<1024xindex> to vector<32x32xindex>

    // Anchor the layout with a store operation
    %ptr = memref.extract_aligned_pointer_as_index %arg0 : memref<32x32xf32> -> index
    %ptr_i64 = arith.index_cast %ptr : index to i64
    %mask = arith.constant dense<true> : vector<32x32xi1>
    %data = arith.constant dense<0.0> : vector<32x32xf32>

    // CHECK: xegpu.store {{.*}} <{{{.*}}layout = #xegpu.layout<inst_data = [32, 32]>{{.*}}}> :
    xegpu.store %data, %ptr_i64[%1], %mask {
      layout = #xegpu.layout<inst_data = [32, 32]>
    } : vector<32x32xf32>, i64, vector<32x32xindex>, vector<32x32xi1>

    gpu.return
  }
}

// -----

// CHECK-LABEL: func.func @dpas_mx_f8e5m2
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<16x64xf8E5M2>, %[[ARG1:[0-9a-zA-Z]+]]: memref<64x32xf8E5M2>, %[[ARG2:[0-9a-zA-Z]+]]: memref<16x32xbf16>
// CHECK-SAME: %[[ARG3:[0-9a-zA-Z]+]]: memref<16x2xf8E8M0FNU>, %[[ARG4:[0-9a-zA-Z]+]]: memref<2x32xf8E8M0FNU>
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} dense<0.000000e+00> : vector<16x32xbf16>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<16x64xf8E5M2> -> !xegpu.tensor_desc<16x64xf8E5M2, #xegpu.layout<inst_data = [8, 32]>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<64x32xf8E5M2> -> !xegpu.tensor_desc<64x32xf8E5M2, #xegpu.layout<inst_data = [32, 16]>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 32]>}> :
// CHECK-SAME: !xegpu.tensor_desc<16x64xf8E5M2, #xegpu.layout<inst_data = [8, 32]>> -> vector<16x64xf8E5M2>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T1]][0, 0] <{layout = #xegpu.layout<inst_data = [32, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<64x32xf8E5M2, #xegpu.layout<inst_data = [32, 16]>> -> vector<64x32xf8E5M2>
// CHECK: %[[T4:.*]] = xegpu.create_nd_tdesc %[[ARG3]] : memref<16x2xf8E8M0FNU> -> !xegpu.tensor_desc<16x2xf8E8M0FNU, #xegpu.layout<inst_data = [8, 1]>>
// CHECK: %[[T5:.*]] = xegpu.load_nd %[[T4]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 1]>}> :
// CHECK-SAME: !xegpu.tensor_desc<16x2xf8E8M0FNU, #xegpu.layout<inst_data = [8, 1]>> -> vector<16x2xf8E8M0FNU>
// CHECK: %[[T6:.*]] = xegpu.create_nd_tdesc %[[ARG4]] : memref<2x32xf8E8M0FNU> -> !xegpu.tensor_desc<2x32xf8E8M0FNU, #xegpu.layout<inst_data = [1, 16]>>
// CHECK: %[[T7:.*]] = xegpu.load_nd %[[T6]][0, 0] <{layout = #xegpu.layout<inst_data = [1, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<2x32xf8E8M0FNU, #xegpu.layout<inst_data = [1, 16]>> -> vector<2x32xf8E8M0FNU>
// CHECK: %[[T8:.*]] = xegpu.dpas_mx %[[T2]], %[[T3]], %[[CST]] scale_a = %[[T5]] scale_b = %[[T7]]
// CHECK-SAME: {layout_a = #xegpu.layout<inst_data = [8, 32]>, layout_a_scale = #xegpu.layout<inst_data = [8, 1]>, layout_b = #xegpu.layout<inst_data = [32, 16]>, layout_b_scale = #xegpu.layout<inst_data = [1, 16]>, layout_cd = #xegpu.layout<inst_data = [8, 16]>} :
// CHECK-SAME: vector<16x64xf8E5M2>, vector<64x32xf8E5M2>, vector<16x32xbf16>, vector<16x2xf8E8M0FNU>, vector<2x32xf8E8M0FNU> -> vector<16x32xbf16>
// CHECK: %[[T9:.*]] = xegpu.create_nd_tdesc %[[ARG2]] : memref<16x32xbf16> -> !xegpu.tensor_desc<16x32xbf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: xegpu.store_nd %[[T8]], %[[T9]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<16x32xbf16>, !xegpu.tensor_desc<16x32xbf16, #xegpu.layout<inst_data = [8, 16]>>
gpu.module @test {
func.func @dpas_mx_f8e5m2(%arg0: memref<16x64xf8E5M2>, %arg1: memref<64x32xf8E5M2>, %arg2: memref<16x32xbf16>,
    %arg3: memref<16x2xf8E8M0FNU>, %arg4: memref<2x32xf8E8M0FNU>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x32xbf16>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<16x64xf8E5M2> -> !xegpu.tensor_desc<16x64xf8E5M2>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<64x32xf8E5M2> -> !xegpu.tensor_desc<64x32xf8E5M2>
  %2 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<16x64xf8E5M2> -> vector<16x64xf8E5M2>
  %3 = xegpu.load_nd %1[0, 0] : !xegpu.tensor_desc<64x32xf8E5M2> -> vector<64x32xf8E5M2>
  %4 = xegpu.create_nd_tdesc %arg3 : memref<16x2xf8E8M0FNU> -> !xegpu.tensor_desc<16x2xf8E8M0FNU>
  %5 = xegpu.load_nd %4[0, 0] : !xegpu.tensor_desc<16x2xf8E8M0FNU> -> vector<16x2xf8E8M0FNU>
  %6 = xegpu.create_nd_tdesc %arg4 : memref<2x32xf8E8M0FNU> -> !xegpu.tensor_desc<2x32xf8E8M0FNU>
  %7 = xegpu.load_nd %6[0, 0] : !xegpu.tensor_desc<2x32xf8E8M0FNU> -> vector<2x32xf8E8M0FNU>
  %8 = xegpu.dpas_mx %2, %3, %cst scale_a = %5 scale_b = %7 : vector<16x64xf8E5M2>, vector<64x32xf8E5M2>, vector<16x32xbf16>, vector<16x2xf8E8M0FNU>, vector<2x32xf8E8M0FNU> -> vector<16x32xbf16>
  %9 = xegpu.create_nd_tdesc %arg2 : memref<16x32xbf16> -> !xegpu.tensor_desc<16x32xbf16>
  xegpu.store_nd %8, %9[0, 0] : vector<16x32xbf16>, !xegpu.tensor_desc<16x32xbf16>
  return
}
}

// -----
// CHECK-LABEL: func.func @dpas_mx_f4e2m1
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<16x128xf4E2M1FN>, %[[ARG1:[0-9a-zA-Z]+]]: memref<128x32xf4E2M1FN>, %[[ARG2:[0-9a-zA-Z]+]]: memref<16x32xbf16>
// CHECK-SAME: %[[ARG3:[0-9a-zA-Z]+]]: memref<16x4xf8E8M0FNU>, %[[ARG4:[0-9a-zA-Z]+]]: memref<4x32xf8E8M0FNU>
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} dense<0.000000e+00> : vector<16x32xbf16>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<16x128xf4E2M1FN> -> !xegpu.tensor_desc<16x128xf4E2M1FN, #xegpu.layout<inst_data = [8, 64]>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<128x32xf4E2M1FN> -> !xegpu.tensor_desc<128x32xf4E2M1FN, #xegpu.layout<inst_data = [64, 16]>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 64]>}> :
// CHECK-SAME: !xegpu.tensor_desc<16x128xf4E2M1FN, #xegpu.layout<inst_data = [8, 64]>> -> vector<16x128xf4E2M1FN>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T1]][0, 0] <{layout = #xegpu.layout<inst_data = [64, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<128x32xf4E2M1FN, #xegpu.layout<inst_data = [64, 16]>> -> vector<128x32xf4E2M1FN>
// CHECK: %[[T4:.*]] = xegpu.create_nd_tdesc %[[ARG3]] : memref<16x4xf8E8M0FNU> -> !xegpu.tensor_desc<16x4xf8E8M0FNU, #xegpu.layout<inst_data = [8, 2]>>
// CHECK: %[[T5:.*]] = xegpu.load_nd %[[T4]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 2]>}> :
// CHECK-SAME: !xegpu.tensor_desc<16x4xf8E8M0FNU, #xegpu.layout<inst_data = [8, 2]>> -> vector<16x4xf8E8M0FNU>
// CHECK: %[[T6:.*]] = xegpu.create_nd_tdesc %[[ARG4]] : memref<4x32xf8E8M0FNU> -> !xegpu.tensor_desc<4x32xf8E8M0FNU, #xegpu.layout<inst_data = [2, 16]>>
// CHECK: %[[T7:.*]] = xegpu.load_nd %[[T6]][0, 0] <{layout = #xegpu.layout<inst_data = [2, 16]>}> :
// CHECK-SAME: !xegpu.tensor_desc<4x32xf8E8M0FNU, #xegpu.layout<inst_data = [2, 16]>> -> vector<4x32xf8E8M0FNU>
// CHECK: %[[T8:.*]] = xegpu.dpas_mx %[[T2]], %[[T3]], %[[CST]] scale_a = %[[T5]] scale_b = %[[T7]]
// CHECK-SAME: {layout_a = #xegpu.layout<inst_data = [8, 64]>, layout_a_scale = #xegpu.layout<inst_data = [8, 2]>, layout_b = #xegpu.layout<inst_data = [64, 16]>, layout_b_scale = #xegpu.layout<inst_data = [2, 16]>, layout_cd = #xegpu.layout<inst_data = [8, 16]>} :
// CHECK-SAME: vector<16x128xf4E2M1FN>, vector<128x32xf4E2M1FN>, vector<16x32xbf16>, vector<16x4xf8E8M0FNU>, vector<4x32xf8E8M0FNU> -> vector<16x32xbf16>
// CHECK: %[[T9:.*]] = xegpu.create_nd_tdesc %[[ARG2]] : memref<16x32xbf16> -> !xegpu.tensor_desc<16x32xbf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK: xegpu.store_nd %[[T8]], %[[T9]][0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<16x32xbf16>, !xegpu.tensor_desc<16x32xbf16, #xegpu.layout<inst_data = [8, 16]>>
gpu.module @test {
func.func @dpas_mx_f4e2m1(%arg0: memref<16x128xf4E2M1FN>, %arg1: memref<128x32xf4E2M1FN>, %arg2: memref<16x32xbf16>,
    %arg3: memref<16x4xf8E8M0FNU>, %arg4: memref<4x32xf8E8M0FNU>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x32xbf16>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<16x128xf4E2M1FN> -> !xegpu.tensor_desc<16x128xf4E2M1FN>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<128x32xf4E2M1FN> -> !xegpu.tensor_desc<128x32xf4E2M1FN>
  %2 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<16x128xf4E2M1FN> -> vector<16x128xf4E2M1FN>
  %3 = xegpu.load_nd %1[0, 0] : !xegpu.tensor_desc<128x32xf4E2M1FN> -> vector<128x32xf4E2M1FN>
  %4 = xegpu.create_nd_tdesc %arg3 : memref<16x4xf8E8M0FNU> -> !xegpu.tensor_desc<16x4xf8E8M0FNU>
  %5 = xegpu.load_nd %4[0, 0] : !xegpu.tensor_desc<16x4xf8E8M0FNU> -> vector<16x4xf8E8M0FNU>
  %6 = xegpu.create_nd_tdesc %arg4 : memref<4x32xf8E8M0FNU> -> !xegpu.tensor_desc<4x32xf8E8M0FNU>
  %7 = xegpu.load_nd %6[0, 0] : !xegpu.tensor_desc<4x32xf8E8M0FNU> -> vector<4x32xf8E8M0FNU>
  %8 = xegpu.dpas_mx %2, %3, %cst scale_a = %5 scale_b = %7 : vector<16x128xf4E2M1FN>, vector<128x32xf4E2M1FN>, vector<16x32xbf16>, vector<16x4xf8E8M0FNU>, vector<4x32xf8E8M0FNU> -> vector<16x32xbf16>
  %9 = xegpu.create_nd_tdesc %arg2 : memref<16x32xbf16> -> !xegpu.tensor_desc<16x32xbf16>
  xegpu.store_nd %8, %9[0, 0] : vector<16x32xbf16>, !xegpu.tensor_desc<16x32xbf16>
  return
}
}
