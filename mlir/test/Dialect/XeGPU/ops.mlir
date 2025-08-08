// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: gpu.module @test {
gpu.module @test {
// CHECK: gpu.func @create_nd_tdesc_1(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @create_nd_tdesc_1(%src: memref<24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  gpu.return
}

// CHECK: gpu.func @create_nd_tdesc_2(%[[arg0:.*]]: ui64, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index, %[[arg4:.*]]: index) {
gpu.func @create_nd_tdesc_2(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  //CHECK: %[[C:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[arg3]],  %[[arg4]]], shape : [%[[arg2]], %[[arg1]]], strides : [%[[arg1]], %[[C]]] : ui64 -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], shape:[%h, %w], strides: [%w, %c1] : ui64 -> !xegpu.tensor_desc<8x16xf32>
  gpu.return
}


// CHECK: gpu.func @create_nd_tdesc_3(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @create_nd_tdesc_3(%src: memref<24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x16xf32, #xegpu.block_tdesc_attr<array_length = 2 : i64>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x16xf32, #xegpu.block_tdesc_attr<array_length = 2>>
  gpu.return
}


// CHECK: gpu.func @create_nd_tdesc_4(%[[arg0:.*]]: memref<2x24x32xf32>) {
gpu.func @create_nd_tdesc_4(%src: memref<2x24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0, 0] : memref<2x24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  gpu.return
}


// CHECK: gpu.func @create_nd_tdesc_5(%[[arg0:.*]]: memref<2x24x32xf32, 3>) {
gpu.func @create_nd_tdesc_5(%src: memref<2x24x32xf32, 3>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<16xf32, #xegpu.block_tdesc_attr<memory_space = slm>>
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<16xf32, #xegpu.block_tdesc_attr<memory_space = slm>>
  gpu.return
}


// CHECK: gpu.func @create_nd_tdesc_6(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @create_nd_tdesc_6(%src: memref<24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x16xf32, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x16xf32, #xegpu.block_tdesc_attr<array_length = 2>>
  gpu.return
}

// CHECK: gpu.func @create_nd_tdesc_7(%[[arg0:.*]]: memref<8x24x32x48x64xf32>) {
gpu.func @create_nd_tdesc_7(%src: memref<8x24x32x48x64xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0, 0, 0, 0] : memref<8x24x32x48x64xf32> -> !xegpu.tensor_desc<8x8x8x24x32xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0, 0, 0] : memref<8x24x32x48x64xf32> -> !xegpu.tensor_desc<8x8x8x24x32xf32>
  gpu.return
}


// CHECK: gpu.func @test_create_nd_tdesc_7(%[[arg0:.*]]: ui64, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index, %[[arg4:.*]]: index, %[[arg5:.*]]: memref<24x32xf32>)
gpu.func @test_create_nd_tdesc_7(%src: ui64, %w : index, %h : index, %x : index, %y : index, %src2: memref<24x32xf32>) {
  //CHECK: %[[C:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg5]] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %3 = xegpu.create_nd_tdesc %src2 : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  gpu.return
}

// CHECK: gpu.func @test_create_nd_tdesc_8(%[[arg0:.*]]: ui64, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index, %[[arg4:.*]]: index)
gpu.func @test_create_nd_tdesc_8(%src: ui64, %w : index, %h : index, %x : index, %y : index) {

  %c1 = arith.constant 1 : index
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0, shape : [%arg2, %arg1], strides : [%arg1, %c1] : ui64 -> !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.create_nd_tdesc %src, shape : [%h, %w], strides : [%w, %c1]  : ui64 -> !xegpu.tensor_desc<8x16xf32>

  gpu.return
}

// CHECK-LABEL: func @test_create_nd_tdesc_9({{.*}})

gpu.func @test_create_nd_tdesc_9(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {

  %c1 = arith.constant 1 : index
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[%arg3, %arg4], shape : [%arg2, %arg1], strides : [%arg1, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], shape:[%h, %w], strides:[%w, %c1]  : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>

  gpu.return
}

// CHECK-LABEL: func @test_create_nd_tdesc_10({{.*}})
gpu.func @test_create_nd_tdesc_10(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0, shape : [%arg2, %arg1], strides : [%arg1, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>
  %2 = xegpu.create_nd_tdesc %src, shape:[%h, %w], strides:[%w, %c1]  : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>

  gpu.return
}

// CHECK: gpu.func @prefetch_nd(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @prefetch_nd(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<8x16xf16>
  gpu.return
}

// CHECK: gpu.func @prefetch_nd_2(%[[arg0:.*]]: memref<48x64xf16>) {
gpu.func @prefetch_nd_2(%src: memref<48x64xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<48x64xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<48x64xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<8x16xf16>
  gpu.return
}

// CHECK: gpu.func @prefetch_nd_offset_1(%[[arg0:.*]]: memref<48x64xf16>,  %arg1: index, %arg2: index) {
gpu.func @prefetch_nd_offset_1(%src: memref<48x64xf16>, %x : index, %y : index) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]] : memref<48x64xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src : memref<48x64xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd %[[R0]][%arg1, %arg2] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1[%x, %y] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<8x16xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd(%[[arg0:.*]]: memref<8x16xf16>) {
gpu.func @subgroup_load_nd(%src: memref<8x16xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, packed}> : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  gpu.return
}

// CHECK: func @simt_load_nd(%[[arg0:.*]]: memref<8x16xf16>) {
gpu.func @simt_load_nd(%src: memref<8x16xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd_2(%[[arg0:.*]]: memref<8x16xf16>) {
gpu.func @subgroup_load_nd_2(%src: memref<8x16xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  gpu.return
}

// CHECK: func @simt_load_nd_2(%[[arg0:.*]]: memref<8x16xf16>) {
gpu.func @simt_load_nd_2(%src: memref<8x16xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16> -> vector<1xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16> -> vector<1xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd_3(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @subgroup_load_nd_3(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  gpu.return
}

// CHECK: func @simt_load_nd_3(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @simt_load_nd_3(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
  gpu.return
}

// CHECK: func @subgroup_load_nd_4(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @subgroup_load_nd_4(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
  gpu.return
}

// CHECK: func @simt_load_nd_4(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_load_nd_4(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd_5(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @subgroup_load_nd_5(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<32xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<32xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<32xf32> -> vector<32xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<32xf32> -> vector<32xf32>
  gpu.return
}

// CHECK: func @simt_load_nd_5(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @simt_load_nd_5(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<32xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<32xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<32xf32> -> vector<2xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<32xf32> -> vector<2xf32>
  gpu.return
}

// CHECK: func @subgroup_load_nd_6(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @subgroup_load_nd_6(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16xf16>
  gpu.return
}

// CHECK: func @simt_load_nd_6(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_load_nd_6(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<32xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> :
    !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<32xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd_7(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @subgroup_load_nd_7(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, packed}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x8x16x2xf16>
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x8x16x2xf16>
  gpu.return
}

// CHECK: func @simt_load_nd_7(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_load_nd_7(%src: memref<24x32xf16>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<32xf16>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> :
    !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<32xf16>
  gpu.return
}

// CHECK: func @subgroup_load_nd_8(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @subgroup_load_nd_8(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8x16xf32>
  gpu.return
}

// CHECK: func @subgroup_load_nd_offset_1(%[[arg0:.*]]: memref<24x32xf32>, %arg1: index, %arg2: index) {
gpu.func @subgroup_load_nd_offset_1(%src: memref<24x32xf32>, %x : index, %y : index) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0 : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]][%arg1, %arg2] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1[%x, %y] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8x16xf32>
  gpu.return
}

// CHECK: func @simt_load_nd_8(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @simt_load_nd_8(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8xf32>
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8xf32>
  gpu.return
}


// CHECK: func @simt_load_nd_offset_1(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @simt_load_nd_offset_1(%src: memref<24x32xf32>) {
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0 : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<16x8xf32>
  // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]][0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8xf32>
  %2 = xegpu.load_nd %1[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8xf32>
  gpu.return
}

// CHECK: func @subgroup_store_nd(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @subgroup_store_nd(%dst: memref<24x32xf16>) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<24x32xf16>
  %1 = arith.constant dense<1.0>: vector<24x32xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  gpu.return
}

// CHECK: func @simt_store_nd(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_store_nd(%src: memref<24x32xf16>) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<48xf16>
  %1 = arith.constant dense<1.0>: vector<48xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  %2 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<48xf16>, !xegpu.tensor_desc<24x32xf16>
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<48xf16>, !xegpu.tensor_desc<24x32xf16>
  gpu.return
}

// CHECK: func @subgroup_store_nd_2(%[[arg0:.*]]: memref<24x32xf16>,  %arg1: index) {
gpu.func @subgroup_store_nd_2(%dst: memref<24x32xf16>, %x : index) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<32xf16>
  %1 = arith.constant dense<1.0>: vector<32xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]] : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  %2 = xegpu.create_nd_tdesc %dst : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]][%arg1] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16>
  xegpu.store_nd %1, %2[%x] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<32xf16>, !xegpu.tensor_desc<32xf16>
  gpu.return
}

// CHECK: func @subgroup_store_nd_offset_1(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @subgroup_store_nd_offset_1(%dst: memref<24x32xf16>) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<32xf16>
  %1 = arith.constant dense<1.0>: vector<32xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16>
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<32xf16>, !xegpu.tensor_desc<32xf16>
  gpu.return
}

// CHECK: func @simt_store_nd_2(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_store_nd_2(%src: memref<24x32xf16>) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<2xf16>
  %1 = arith.constant dense<1.0>: vector<2xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  %2 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<32xf16>
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<2xf16>, !xegpu.tensor_desc<32xf16>
  gpu.return
}

// CHECK: func @simt_store_nd_offset_1(%[[arg0:.*]]: memref<24x32xf16>) {
gpu.func @simt_store_nd_offset_1(%src: memref<24x32xf16>) {
  // CHECK: %[[C:.*]] = arith.constant dense<1.000000e+00> : vector<2xf16>
  %1 = arith.constant dense<1.0>: vector<2xf16>
  // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0 : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  %2 = xegpu.create_nd_tdesc %src : memref<24x32xf16> -> !xegpu.tensor_desc<32xf16>
  // CHECK: xegpu.store_nd %[[C]], %[[R0]][0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<32xf16>
  xegpu.store_nd %1, %2[0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<2xf16>, !xegpu.tensor_desc<32xf16>
  gpu.return
}

// CHECK: gpu.func @update_nd_tdesc(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @update_nd_tdesc(%src: memref<24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  // CHECK: %[[R1:.*]] = xegpu.update_nd_offset %[[REG]], [0, 16] : !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.update_nd_offset %1, [0, 16]: !xegpu.tensor_desc<8x16xf32>
  gpu.return
}

// CHECK: gpu.func @update_nd_tdesc_2(%[[arg0:.*]]: memref<8x24x32xf32>) {
gpu.func @update_nd_tdesc_2(%src: memref<8x24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0, 0] : memref<8x24x32xf32> -> !xegpu.tensor_desc<2x8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<8x24x32xf32> -> !xegpu.tensor_desc<2x8x16xf32>
  // CHECK: %[[R1:.*]] = xegpu.update_nd_offset %[[REG]], [0, 0, 16] : !xegpu.tensor_desc<2x8x16xf32>
  %2 = xegpu.update_nd_offset %1, [0, 0, 16]: !xegpu.tensor_desc<2x8x16xf32>
  gpu.return
}

// CHECK: gpu.func @create_tdesc(%[[arg0:.*]]: ui64) {
gpu.func @create_tdesc(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex>  -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}


// CHECK: gpu.func @create_tdesc_1(%[[arg0:.*]]: memref<?xf32, 3>) {
gpu.func @create_tdesc_1(%src: memref<?xf32, 3>) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : memref<?xf32, 3>, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32, 3>, vector<4xindex>  -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 2>>
  gpu.return
}


// CHECK: gpu.func @create_tdesc_2(%[[arg0:.*]]: memref<?xf32>) {
gpu.func @create_tdesc_2(%src: memref<?xf32>) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : memref<?xf32>, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>  -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  gpu.return
}


// CHECK: gpu.func @create_tdesc_3(%[[arg0:.*]]: ui64) {
gpu.func @create_tdesc_3(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}

// CHECK: gpu.func @create_tdesc_4(%[[arg0:.*]]: ui64) {
gpu.func @create_tdesc_4(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : vector<2x4xindex>
  %0 = arith.constant dense<[[0, 8, 16, 24], [32, 40, 48, 56]]> : vector<2x4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}


// CHECK: gpu.func @subgroup_load(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_load(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1> -> vector<4x2xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1> -> vector<4x2xf32>
  gpu.return
}

// CHECK: gpu.func @simt_load(%[[arg0:.*]]: ui64) {
gpu.func @simt_load(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1> -> vector<2xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1> -> vector<2xf32>
  gpu.return
}

// CHECK: gpu.func @subgroup_load_2(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_load_2(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1> -> vector<4xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1> -> vector<4xf32>
  gpu.return
}

// CHECK: gpu.func @simt_load_2(%[[arg0:.*]]: ui64) {
gpu.func @simt_load_2(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1> -> vector<1xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1> -> vector<1xf32>
  gpu.return
}

// CHECK: gpu.func @subgroup_load_3(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_load_3(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<4xi1> -> vector<4x8xf16>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<4xi1> -> vector<4x8xf16>
  gpu.return
}

// CHECK: gpu.func @simt_load_3(%[[arg0:.*]]: ui64) {
gpu.func @simt_load_3(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<4xi1> -> vector<8xf16>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<4xi1> -> vector<8xf16>
  gpu.return
}

// CHECK: gpu.func @subgroup_load_4(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_load_4(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : vector<2x4xindex>
  %0 = arith.constant dense<[[0, 8, 16, 24], [32, 40, 48, 56]]> : vector<2x4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<2x4xi1>
  %1 = arith.constant dense<1>: vector<2x4xi1>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  //CHECK: %[[R1:.*]] = xegpu.load %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<2x4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<2x4xi1> -> vector<2x4x8xf16>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<2x4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<2x4xi1> -> vector<2x4x8xf16>
  gpu.return
}

// CHECK: gpu.func @subgroup_load_offset_1(%arg0: memref<?xf16>) {
gpu.func @subgroup_load_offset_1(%src: memref<?xf16>) {
  %offset = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[R1:.*]] = xegpu.load %arg0[%cst], %cst_0 <{chunk_size = 2 : i64, l1_hint = #xegpu.cache_hint<cached>}> : memref<?xf16>, vector<4xindex>, vector<4xi1> -> vector<4x2xf16>
  %val = xegpu.load %src[%offset], %mask <{chunk_size=2, l1_hint = #xegpu.cache_hint<cached>}>
      : memref<?xf16>, vector<4xindex>, vector<4xi1> -> vector<4x2xf16>
  gpu.return
}

// CHECK: gpu.func @subgroup_store(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_store(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant dense<2.900000e+00> : vector<4x2xf32>
  %2 = arith.constant dense<2.9>: vector<4x2xf32>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4x2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4x2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @simt_store(%[[arg0:.*]]: ui64) {
gpu.func @simt_store(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant dense<2.900000e+00> : vector<2xf32>
  %2 = arith.constant dense<2.9>: vector<2xf32>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @subgroup_store_2(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_store_2(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant {{.*}} : vector<4x2xf16>
  %2 = arith.constant dense<2.9>: vector<4x2xf16>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4x2xf16>, !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4x2xf16>, !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @simt_store_2(%[[arg0:.*]]: ui64) {
gpu.func @simt_store_2(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant {{.*}} : vector<2xf16>
  %2 = arith.constant dense<2.9>: vector<2xf16>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @subgroup_store_3(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_store_3(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant {{.*}} : vector<4xf32>
  %2 = arith.constant dense<2.9>: vector<4xf32>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4xf32>, !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<4xf32>, !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @simt_store_3(%[[arg0:.*]]: ui64) {
gpu.func @simt_store_3(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<4xi1>
  %1 = arith.constant dense<1>: vector<4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant dense<2.900000e+00> : vector<1xf32>
  %2 = arith.constant dense<2.9>: vector<1xf32>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @subgroup_store_4(%[[arg0:.*]]: ui64) {
gpu.func @subgroup_store_4(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : vector<2x4xindex>
  %0 = arith.constant dense<[[0, 8, 16, 24], [32, 40, 48, 56]]> : vector<2x4xindex>
  //CHECK: %[[cst1:.*]] = arith.constant dense<true> : vector<2x4xi1>
  %1 = arith.constant dense<1>: vector<2x4xi1>
  //CHECK: %[[cst2:.*]] = arith.constant {{.*}} : vector<2x4xf32>
  %2 = arith.constant dense<2.9>: vector<2x4xf32>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4xf32, #xegpu.scatter_tdesc_attr<>>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<2x4xindex> -> !xegpu.tensor_desc<2x4xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: xegpu.store %[[cst2]], %[[R0]], %[[cst1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2x4xf32>, !xegpu.tensor_desc<2x4xf32, #xegpu.scatter_tdesc_attr<>>, vector<2x4xi1>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2x4xf32>, !xegpu.tensor_desc<2x4xf32, #xegpu.scatter_tdesc_attr<>>, vector<2x4xi1>
  gpu.return
}

// CHECK: gpu.func @subgroup_store_offset_1(%arg0: memref<?xf16>) {
gpu.func @subgroup_store_offset_1(%dest: memref<?xf16>) {
  %val = arith.constant dense<2.9>: vector<4x2xf16>
  %offset = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  //CHECK: xegpu.store %[[R0:.*]], %arg0[%cst_0], %cst_1 <{chunk_size = 2 : i64, l1_hint = #xegpu.cache_hint<cached>}> : vector<4x2xf16>, memref<?xf16>, vector<4xindex>, vector<4xi1>
  xegpu.store %val, %dest[%offset], %mask <{chunk_size=2, l1_hint = #xegpu.cache_hint<cached>}>
      : vector<4x2xf16>, memref<?xf16>, vector<4xindex>, vector<4xi1>
  gpu.return
}

// CHECK: gpu.func @prefetch(%[[arg0:.*]]: ui64) {
gpu.func @prefetch(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: xegpu.prefetch %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}

// CHECK: gpu.func @prefetch_offset(%[[arg0:.*]]: ui64) {
gpu.func @prefetch_offset(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // CHECK: xegpu.prefetch %[[arg0]][%cst] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : ui64, vector<4xindex>
  xegpu.prefetch %src[%0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: ui64, vector<4xindex>
  gpu.return
}

// CHECK: gpu.func @create_update_tdesc(%[[arg0:.*]]: ui64) {
gpu.func @create_update_tdesc(%src: ui64) {
  //CHECK: %[[cst:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst]] : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  //CHECK: %[[st:.*]] = arith.constant dense<32> : vector<4xindex>
  //CHECK: %[[R1:.*]] = xegpu.update_offset %[[R0]], %[[st]] : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<4xindex>
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  %s = arith.constant dense<[32, 32, 32, 32]> : vector<4xindex>
  %2 = xegpu.update_offset %1, %s : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xindex>
  gpu.return
}

// CHECK: gpu.func @subgroup_dpas(%[[arg0:.*]]: vector<8x16xf16>, %[[arg1:.*]]: vector<16x16xf16>)
gpu.func @subgroup_dpas(%a : vector<8x16xf16>, %b: vector<16x16xf16>) {
  // CHECK: %0 = xegpu.dpas %[[arg0]], %[[arg1]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %1 = xegpu.dpas %a, %b: vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  gpu.return
}

// CHECK: gpu.func @simt_dpas(%[[arg0:.*]]: vector<8xf16>, %[[arg1:.*]]: vector<16xf16>)
gpu.func @simt_dpas(%a : vector<8xf16>, %b: vector<16xf16>) {
  // CHECK: xegpu.dpas %[[arg0]], %[[arg1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
  %1 = xegpu.dpas %a, %b : vector<8xf16>, vector<16xf16> -> vector<8xf32>
  gpu.return
}

// CHECK: gpu.func @subgroup_dpas_packed_b(%[[arg0:.*]]: vector<8x16xf16>, %[[arg1:.*]]: vector<8x16x2xf16>)
gpu.func @subgroup_dpas_packed_b(%a : vector<8x16xf16>, %b: vector<8x16x2xf16>) {
  // CHECK: %0 = xegpu.dpas %[[arg0]], %[[arg1]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  %1 = xegpu.dpas %a, %b: vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  gpu.return
}

// CHECK: gpu.func @subgroup_atomic_rmw(%[[arg0:.*]]: ui64, %[[arg1:.*]]: vector<16xf32>, %[[arg2:.*]]: vector<16xi1>)
gpu.func @subgroup_atomic_rmw(%src: ui64, %value : vector<16xf32>, %mask : vector<16xi1>) {
  //CHECK: %[[c:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  //CHECK: %[[R0:.*]] = xegpu.create_tdesc %[[arg0]], %[[c]] : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src, %c: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  //CHECK: %[[R1:.*]] = xegpu.atomic_rmw addf %[[R0]], %[[arg2]], %[[arg1]] : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
  xegpu.atomic_rmw addf %1, %mask, %value: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
  gpu.return
}

// CHECK: gpu.func @alloc_nbarrier({{.*}}) {
gpu.func @alloc_nbarrier() {
  // CHECK: xegpu.alloc_nbarrier
  xegpu.alloc_nbarrier 8
  gpu.return
}

// CHECK: gpu.func @init_nbarrier({{.*}}) {
gpu.func @init_nbarrier() {
  //CHECK: %[[c1:.*]] = arith.constant 1 : i8
  //CHECK: %[[c16:.*]] = arith.constant 16 : i8
  %nbarrier_id = arith.constant 1 : i8
  %threads_count = arith.constant 16 : i8
  //CHECK: xegpu.init_nbarrier %[[c1]], %[[c16]] : i8, i8 -> !xegpu.nbarrier
  %nbarrier = xegpu.init_nbarrier %nbarrier_id, %threads_count : i8, i8 -> !xegpu.nbarrier
  gpu.return
}

// CHECK: gpu.func @nbarrier_arrive(%[[arg0:.*]]: !xegpu.nbarrier) {
gpu.func @nbarrier_arrive(%nbarrier : !xegpu.nbarrier) {
  //CHECK: xegpu.nbarrier_arrive %[[arg0]] : !xegpu.nbarrier
  xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
  gpu.return
}

// CHECK: gpu.func @nbarrier_wait(%[[arg0:.*]]: !xegpu.nbarrier) {
gpu.func @nbarrier_wait(%nbarrier : !xegpu.nbarrier) {
  //CHECK: xegpu.nbarrier_wait %[[arg0]] : !xegpu.nbarrier
  xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
  gpu.return
}

// CHECK-LABEL: gpu.func @fence({{.*}}) {
gpu.func @fence() {
  //CHECK: xegpu.fence memory_kind = global, fence_scope = workgroup
  xegpu.fence memory_kind = global, fence_scope = workgroup
  gpu.return
}

}
