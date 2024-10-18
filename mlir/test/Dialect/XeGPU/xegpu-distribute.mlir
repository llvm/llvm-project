// RUN: mlir-opt -test-xegpu-distribute -split-input-file %s | FileCheck %s

#sg_map_16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#blk_tdesc = #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>

// CHECK-LABEL: test_store_nd_distribution
// CHECK: %[[laneid:.*]] = gpu.lane_id
// CHECK: %[[res:.*]]:2 = vector.warp_execute_on_lane_0(%[[laneid]])[16] args(%{{.*}}, %{{.*}} :  vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>)
// CHECK-SAME: -> (!xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>, vector<24x2xf16>)
// CHECK: ^bb0(%[[src:.*]]: vector<24x32xf16>, %[[dst:.*]]: !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>)
// CHECK: vector.yield %[[dst]], %[[src]] : !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>, vector<24x32xf16>
// CHECK: xegpu.store_nd %[[res]]#1, %[[res]]#0 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> :
// CHECK-SAME: vector<24x2xf16>, !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>

func.func @test_store_nd_distribution(%src: vector<24x32xf16>, %dst: !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>) -> () {
  %laneid = gpu.lane_id
  vector.warp_execute_on_lane_0(%laneid)[16]
        args(%src, %dst: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>) {
    ^bb0(%arg0: vector<24x32xf16>, %arg1: !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>):
    xegpu.store_nd %arg0, %arg1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>
  }
  return
}

// -----

#sg_map_16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#blk_tdesc = #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>

// CHECK-LABEL: test_load_nd_distribution
// CHECK: %[[laneid:.*]] = gpu.lane_id
// CHECK: %[[res:.*]]:2 = vector.warp_execute_on_lane_0(%[[laneid]])[16] args(%{{.*}} :  !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>)
// CHECK-SAME: -> (vector<24x2xf16>, !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>)
// CHECK: ^bb0(%[[dst:.*]]: !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>)
// CHECK: %[[dead:.*]] = xegpu.load_nd
// CHECK: vector.yield %[[dead]], %[[dst]] : vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
// CHECK: %[[load:.*]] = xegpu.load_nd %[[res]]#1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> :
// CHECK-SAME: !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<24x2xf16>
// CHECK: return %[[load]]

func.func @test_load_nd_distribution(%dst: !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>) -> (vector<24x2xf16>) {
  %laneid = gpu.lane_id
  %r = vector.warp_execute_on_lane_0(%laneid)[16]
        args(%dst: !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>) -> (vector<24x2xf16>) {
    ^bb0(%arg0: !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>):
    %0 = xegpu.load_nd %arg0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16> -> vector<24x32xf16>
    vector.yield %0 : vector<24x32xf16>
  }
  return %r : vector<24x2xf16>
}

// -----

#sg_map_16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#blk_tdesc = #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>

// CHECK-LABEL: test_create_nd_desc_distribution
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[laneid:.*]] = gpu.lane_id
// CHECK: %[[res:.*]]:2 = vector.warp_execute_on_lane_0(%[[laneid]])[16] args(%{{.*}} : memref<24x32xf16>)
// CHECK-SAME: -> (!xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>, memref<24x32xf16>)
// CHECK: ^bb0(%[[dst:.*]]: memref<24x32xf16>)
// CHECK: %[[dead:.*]] = xegpu.create_nd_tdesc
// CHECK: vector.yield %[[dead]], %[[dst]] :
// CHECK-SAME: !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>, memref<24x32xf16>
// CHECK: %[[view:.*]] = memref.subview %[[res]]#1[%[[C0]], %[[laneid]]] [24, 2] [1, 1] : memref<24x32xf16> to memref<24x2xf16, strided<[32, 1], offset: ?>>
// CHECK: %[[desc:.*]] = xegpu.create_nd_tdesc %[[view]][0, 0] : memref<24x2xf16, strided<[32, 1], offset: ?>>
// CHECK-SAME: -> !xegpu.tensor_desc<24x2xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
// CHECK: return %[[desc]]

func.func @test_create_nd_desc_distribution(%dst: memref<24x32xf16>) -> (!xegpu.tensor_desc<24x2xf16, #blk_tdesc, #sg_map_16>) {
  %laneid = gpu.lane_id
  %r = vector.warp_execute_on_lane_0(%laneid)[16]
        args(%dst: memref<24x32xf16>) -> (!xegpu.tensor_desc<24x2xf16, #blk_tdesc, #sg_map_16>) {
    ^bb0(%arg0: memref<24x32xf16>):
    %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>
    vector.yield %0 : !xegpu.tensor_desc<24x32xf16, #blk_tdesc, #sg_map_16>
  }
  return %r : !xegpu.tensor_desc<24x2xf16, #blk_tdesc, #sg_map_16>
}
