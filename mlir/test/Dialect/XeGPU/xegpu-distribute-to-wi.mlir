// RUN: mlir-opt -xegpu-distribute-to-wi -split-input-file %s | FileCheck %s

#sg_map_16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

// CHECK-LABEL: test
gpu.module @test {
// CHECK-LABEL: test_load_store_nd_distribution
// CHECK: (%[[arg0:.*]]: memref<24x32xf16>, %[[arg1:.*]]: memref<24x32xf16>) {
func.func @test_load_store_nd_distribution(%src: memref<24x32xf16>, %dst: memref<24x32xf16>) -> () {
// CHECK: %[[C1:.*]] = arith.constant
// CHECK: %[[ID:.*]] = gpu.lane_id
// CHECK: %[[V0:.*]] = memref.subview %[[arg0]][%[[ID]], %[[ID]]] [24, 2] [1, 1] : memref<24x32xf16> to memref<24x2xf16, strided<[32, 1], offset: ?>>
// CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[V0]][%[[ID]], %[[ID]]] : memref<24x2xf16, strided<[32, 1], offset: ?>> -> !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #sg_map_16>
// CHECK: %[[V1:.*]] = memref.subview %[[arg1]][%[[ID]], %[[ID]]] [24, 2] [1, 1] : memref<24x32xf16> to memref<24x2xf16, strided<[32, 1], offset: ?>>
// CHECK: %[[R1:.*]] = xegpu.create_nd_tdesc %[[V1]][%[[ID]], %[[ID]]] : memref<24x2xf16, strided<[32, 1], offset: ?>> -> !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #sg_map_16>
// CHECK: %[[L0:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> :
// CHECK-SAME: !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<24x2xf16>
  %2 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<24x32xf16, #sg_map_16> -> vector<24x32xf16>
  %3 = arith.constant dense<1.0>: vector<24x32xf16>
// CHECK: %[[A0:.*]] = arith.addf %[[L0]], %[[C1]] : vector<24x2xf16>
  %4 = arith.addf %2, %3: vector<24x32xf16>
// CHECK: xegpu.store_nd %[[A0]], %[[R1]]
  xegpu.store_nd %4, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #sg_map_16>
  return
}
}

