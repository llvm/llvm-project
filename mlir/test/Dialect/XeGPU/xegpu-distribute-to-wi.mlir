// RUN: mlir-opt -xegpu-distribute-to-wi -split-input-file %s | FileCheck %s

#sg_map_16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

// CHECK-LABEL: test
gpu.module @test {
// CHECK-LABEL: test_load_store_nd_distribution
// CHECK: (%[[arg0:.*]]: memref<24x32xf16>, %[[arg1:.*]]: memref<24x32xf16>) {
func.func @test_load_store_nd_distribution(%src: memref<24x32xf16>, %dst: memref<24x32xf16>) -> () {
// CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #sg_map_16>
// CHECK: %[[R1:.*]] = xegpu.create_nd_tdesc %[[arg1]][0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #sg_map_16>
// CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> :
// CHECK-SAME: !xegpu.tensor_desc<24x32xf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<24x2xf16>
  %2 = xegpu.load_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<24x32xf16, #sg_map_16> -> vector<24x32xf16>
  %3 = arith.constant dense<1.0>: vector<24x32xf16>
  %4 = arith.addf %2, %3: vector<24x32xf16>
  xegpu.store_nd %4, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #sg_map_16>
  return
}
}

// -----
