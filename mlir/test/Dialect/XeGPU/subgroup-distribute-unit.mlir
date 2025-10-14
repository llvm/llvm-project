// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=pvc' -test-xegpu-sg-distribute -allow-unregistered-dialect \
// RUN: -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @store_nd_1d
// CHECK:         (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:         %[[W:.*]]:3 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16]
// CHECK-SAME:      -> (vector<1xf32>, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index) {
// CHECK:           gpu.yield %{{.*}} : vector<16xf32>,
// CHECK-SAME:        !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#1 : !xegpu.tensor_desc<16xf32,
// CHECK-SAME:      #xegpu.layout<lane_layout = [16], lane_data = [1]>> to !xegpu.tensor_desc<16xf32> {resolve_simt_type_mismatch}
// CHECK-NEXT:    xegpu.store_nd %[[W]]#0, %[[T1]][%[[W]]#2]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
gpu.module @xevm_module{
  gpu.func @store_nd_1d(%laneid: index) {
    %c0 = arith.constant 0 : index
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
      %cst = "some_op"() : () -> vector<16xf32>
      xegpu.store_nd %cst, %0 [%c0] {layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @store_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16]
// CHECK-SAME:    -> (vector<16x1xf16>, !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:    #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index) {
// CHECK:         gpu.yield %{{.*}} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[CAST:.*]] = vector.shape_cast %[[W]]#0 : vector<16x1xf16> to vector<16xf16>
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#1 : !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:    #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> to !xegpu.tensor_desc<16x16xf16> {resolve_simt_type_mismatch}
// CHECK-NEXT:  xegpu.store_nd %[[CAST]], %[[T1]][%[[W]]#2, %[[W]]#3]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @xevm_module{
  gpu.func @store_nd_2d(%laneid : index) {
    %c0 = arith.constant 0 : index
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      %cst = "some_op"() : () -> vector<16x16xf16>
      xegpu.store_nd %cst, %0 [%c0, %c0] {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    }
    gpu.return
  }
}



// -----
// CHECK-LABEL: gpu.func @load_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:3 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] -> (vector<1xf32>,
// CHECK-SAME:    !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index) {
// CHECK:         gpu.yield %{{.*}} : vector<16xf32>, !xegpu.tensor_desc<16xf32,
// CHECK-SAME:      #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#1 : !xegpu.tensor_desc<16xf32,
// CHECK-SAME:    #xegpu.layout<lane_layout = [16], lane_data = [1]>> to !xegpu.tensor_desc<16xf32> {resolve_simt_type_mismatch}
// CHECK-NEXT:  xegpu.load_nd %[[T1]][%[[W]]#2]  : !xegpu.tensor_desc<16xf32> -> vector<1xf32>
gpu.module @xevm_module{
  gpu.func @load_nd_1d(%laneid: index) {
    %c0 = arith.constant 0 : index
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<1xf32>) {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
      %1 = xegpu.load_nd %0 [%c0]  {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} :
        !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>> -> vector<16xf32>
      gpu.yield %1 : vector<16xf32>
    }
    "some_user_op"(%r) : (vector<1xf32>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] -> (vector<16x1xf16>, !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index) {
// CHECK:         gpu.yield %{{.*}} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#1 : !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:     #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> to !xegpu.tensor_desc<16x16xf16> {resolve_simt_type_mismatch}
// CHECK-NEXT:  %[[T2:.*]] = xegpu.load_nd %[[T1]][%[[W]]#2, %[[W]]#3]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK:       vector.shape_cast %[[T2]] : vector<16xf16> to vector<16x1xf16>
gpu.module @xevm_module{
  gpu.func @load_nd_2d(%laneid: index) {
    %c0 = arith.constant 0 : index
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<16x1xf16>) {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      %1 = xegpu.load_nd %0[%c0, %c0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
      gpu.yield %1 : vector<16x16xf16>
    }
    "some_user_op"(%r) : (vector<16x1xf16>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_nd_array_length
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] -> (vector<2x16x1xf16>,
// CHECK-SAME:    !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>,
// CHECK-SAME:    #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index) {
// CHECK:         gpu.yield %{{.*}} : vector<2x16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<
// CHECK-SAME:      array_length = 2 : i64>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#1 : !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.block_tdesc_attr<array_length = 2 : i64>, #xegpu.layout<lane_layout = [1, 16],
// CHECK-SAME:      lane_data = [1, 1]>> to !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
// CHECK-NEXT:  %[[T2:.*]] = xegpu.load_nd %[[T1]][%[[W]]#2, %[[W]]#3]  : !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:    #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<32xf16>
// CHECK-NEXT:  vector.shape_cast %[[T2]] : vector<32xf16> to vector<2x16x1xf16>
gpu.module @xevm_module{
  gpu.func @load_nd_array_length(%laneid: index) {
    %c0 = arith.constant 0 : index
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2x16x1xf16>) {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>,
        #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      %1 = xegpu.load_nd %0[%c0, %c0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>,
          #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<2x16x16xf16>
      gpu.yield %1 : vector<2x16x16xf16>
    }
    "some_user_op"(%r) : (vector<2x16x1xf16>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @dpas
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] ->
// CHECK-SAME:    (vector<8x1xf32>, vector<8x1xf16>, vector<16x1xf16>, vector<8x1xf32>) {
// CHECK:         gpu.yield %{{.*}} : vector<8x16xf32>, vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>
// CHECK-NEXT:  }
// CHECK-DAG:   %[[T1:.*]] = vector.shape_cast %[[W]]#1 : vector<8x1xf16> to vector<8xf16>
// CHECK-DAG:   %[[T2:.*]] = vector.shape_cast %[[W]]#2 : vector<16x1xf16> to vector<16xf16>
// CHECK-DAG:   %[[T3:.*]] = vector.shape_cast %[[W]]#3 : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT:  %[[T4:.*]] = xegpu.dpas %[[T1]], %[[T2]], %[[T3]] : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
// CHECK-NEXT:  vector.shape_cast %[[T4]] : vector<8xf32> to vector<8x1xf32>
gpu.module @xevm_module{
  gpu.func @dpas(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<8x1xf32>) {
      %0 = "some_op"() : () -> vector<8x16xf16>
      %1 = "some_op"() : () -> vector<16x16xf16>
      %2 = "some_op"() : () -> vector<8x16xf32>
      %3 = xegpu.dpas %0, %1, %2
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
          layout_operand_2 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      gpu.yield %3 : vector<8x16xf32>
    }
    "some_user_op"(%r) : (vector<8x1xf32>) -> ()
    gpu.return
  }
}


// -----
// CHECK-LABEL: gpu.func @create_nd_tdesc_non_memref
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: ui64, %[[ARG1:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%[[ARG1]])[16] -> (!xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, ui64) {
// CHECK:         gpu.yield %{{.*}} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, ui64
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = xegpu.create_nd_tdesc %[[W]]#1, shape : [64, 128], strides : [128, 1] : ui64 -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT:  builtin.unrealized_conversion_cast %[[T1]] : !xegpu.tensor_desc<16x16xf16> to !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:    #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> {resolve_simt_type_mismatch}
gpu.module @xevm_module{
  gpu.func @create_nd_tdesc_non_memref(%arg0: ui64, %laneid: index) {
    %c0 = arith.constant 0 : index
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (!xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
      %0 = xegpu.create_nd_tdesc %arg0, shape:[64, 128], strides:[128, 1] : ui64 ->
        !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.yield %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    }
    "some_user_op"(%r)
      : (!xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @prefetch_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:3 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] -> (!xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, index, index) {
// CHECK:         gpu.yield %{{.*}} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-SAME:      , index, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#0 : !xegpu.tensor_desc<16x16xf16,
// CHECK-SAME:    #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> to !xegpu.tensor_desc<16x16xf16> {resolve_simt_type_mismatch}
// CHECK-NEXT:  xegpu.prefetch_nd %[[T1]][%[[W]]#1, %[[W]]#2]
// CHECK-SAME:    <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16>
gpu.module @xevm_module{
  gpu.func @prefetch_2d(%laneid: index) {
    %c0 = arith.constant 0 : index
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %0 = "some_op"() : ()
        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      xegpu.prefetch_nd %0[%c0, %c0]
        <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @prefetch_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: index) {
// CHECK:       %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%[[ARG0]])[16] -> (!xegpu.tensor_desc<16xf16,
// CHECK-SAME:     #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index) {
// CHECK:         gpu.yield %{{.*}} : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>, index
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]]#0 : !xegpu.tensor_desc<16xf16,
// CHECK-SAME:    #xegpu.layout<lane_layout = [16], lane_data = [1]>> to !xegpu.tensor_desc<16xf16> {resolve_simt_type_mismatch}
// CHECK-NEXT:  xegpu.prefetch_nd %[[T1]][%[[W]]#1] <{l1_hint = #xegpu.cache_hint<cached>,
// CHECK-SAME:    l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16>
gpu.module @xevm_module{
  gpu.func @prefetch_1d(%laneid: index) {
    %c0 = arith.constant 0 : index
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %0 = "some_op"() : ()
        -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
      xegpu.prefetch_nd %0[%c0]
        <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @gpu_barrier({{.*}}) {
// CHECK:  gpu.warp_execute_on_lane_0(%{{.*}})[16] -> ({{.*}}) {
// CHECK:     gpu.yield %{{.*}}
// CHECK:  }
// CHECK:  %{{.*}} = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16xf16> -> vector<1xf16>
// CHECK:  gpu.barrier
gpu.module @xevm_module{
  gpu.func @gpu_barrier(%laneid: index) {
    %c0 = arith.constant 0 : index
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<1xf16>) {
      %0 = "some_op"() : () -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
      %1 = xegpu.load_nd %0[%c0]
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>> -> vector<16xf16>
      gpu.barrier
      gpu.yield %1 : vector<16xf16>
    }
    "some_user_op"(%r) : (vector<1xf16>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim0_reduction
// CHECK:       %[[ACC:.*]] = arith.constant {{.*}} dense<0.000000e+00> : vector<32xf32>
// CHECK:       %[[W:.*]]:3 = gpu.warp_execute_on_lane_0(%{{.*}})[16]
// CHECK-SAME:    -> (vector<2xf32>, vector<16x2xf32>, vector<2xf32>) {
// CHECK:         %[[SRC:.*]] = "some_def"() {{.*}} : () -> vector<16x32xf32>
// CHECK:         gpu.yield %{{.*}}, %[[SRC]], %[[ACC]] : vector<32xf32>, vector<16x32xf32>, vector<32xf32>
// CHECK-NEXT:  }
// CHECK:       %[[T1:.*]] = vector.extract_strided_slice %[[W]]#1
// CHECK-SAME:    {offsets = [0, 0], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK:       %[[T2:.*]] = vector.shape_cast %[[T1]] : vector<16x1xf32> to vector<16xf32>
// CHECK:       %[[T3:.*]] = vector.extract %[[W]]#2[0] : f32 from vector<2xf32>
// CHECK:       %[[T4:.*]] = vector.reduction <add>, %[[T2]], %[[T3]] : vector<16xf32> into f32
// CHECK:       %[[T5:.*]] = vector.extract_strided_slice %[[W]]#1
// CHECK-SAME:    {offsets = [0, 1], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK:       %[[T6:.*]] = vector.shape_cast %[[T5]] : vector<16x1xf32> to vector<16xf32>
// CHECK:       %[[T7:.*]] = vector.extract %[[W]]#2[1] : f32 from vector<2xf32>
// CHECK:       %[[T8:.*]] = vector.reduction <add>, %[[T6]], %[[T7]] : vector<16xf32> into f32
// CHECK:       %[[T9:.*]] = vector.from_elements %[[T4]], %[[T8]] : vector<2xf32>
gpu.module @xevm_module{
gpu.func @vector_multi_reduction_dim1_distributed_dim0_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
  %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2xf32>) {
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : () -> (vector<16x32xf32>)
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
      dense<0.0>  : vector<32xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
    {
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_operand_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>,
      layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>
    }  [0]
    : vector<16x32xf32> to vector<32xf32>
    gpu.yield %1 : vector<32xf32>
  }
  "some_user_op"(%r) : (vector<2xf32>) -> ()
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction
// CHECK:      %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<2xf32>) {
// CHECK-NEXT:   %[[SRC:.*]] = "some_def"() {{.*}} : () -> vector<2x16xf32>
// CHECK-NEXT:   %[[T2:.*]] = vector.extract %[[SRC]][0] : vector<16xf32> from vector<2x16xf32>
// CHECK-NEXT:   %[[T3:.*]] = vector.reduction <add>, %[[T2]], %cst : vector<16xf32> into f32
// CHECK-NEXT:   %[[T4:.*]] = vector.extract %[[SRC]][1] : vector<16xf32> from vector<2x16xf32>
// CHECK-NEXT:   %[[T5:.*]] = vector.reduction <add>, %[[T4]], %cst : vector<16xf32> into f32
// CHECK-NEXT:   %[[T6:.*]] = vector.from_elements %[[T3]], %[[T5]] : vector<2xf32>
// CHECK-NEXT:   gpu.yield %[[T6]] : vector<2xf32>
// CHECK-NEXT: }
gpu.module @xevm_module{
gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
  %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2xf32>) {
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : () -> (vector<2x16xf32>)
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
      dense<0.0>  : vector<2xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
        layout_operand_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>,
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>
      }
      [1] : vector<2x16xf32> to vector<2xf32>
    gpu.yield %1 : vector<2xf32>
  }
  "some_user_op"(%r) : (vector<2xf32>) -> ()
  gpu.return
}
}

// -----
// CHECK-LABEL:   gpu.func @vector_multi_reduction_dim0_distributed_dim1_reduction
// CHECK:       %[[ACC:.*]] = arith.constant {{.*}} dense<0.000000e+00> : vector<32xf32>
// CHECK:       %[[W:.*]]:3 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<2xf32>, vector<2x16xf32>, vector<2xf32>) {
// CHECK:         %[[SRC:.*]] = "some_def"() {{.*}} : () -> vector<32x16xf32>
// CHECK:         gpu.yield %9, %[[SRC]], %[[ACC]] : vector<32xf32>, vector<32x16xf32>, vector<32xf32>
// CHECK:       }
// CHECK:       %[[T1:.*]] = vector.extract %[[W]]#1[0] : vector<16xf32> from vector<2x16xf32>
// CHECK:       %[[T2:.*]] = vector.extract %[[W]]#2[0] : f32 from vector<2xf32>
// CHECK:       %[[T3:.*]] = vector.reduction <add>, %[[T1]], %[[T2]] : vector<16xf32> into f32
// CHECK:       %[[T4:.*]] = vector.extract %[[W]]#1[1] : vector<16xf32> from vector<2x16xf32>
// CHECK:       %[[T5:.*]] = vector.extract %[[W]]#2[1] : f32 from vector<2xf32>
// CHECK:       %[[T6:.*]] = vector.reduction <add>, %[[T4]], %[[T5]] : vector<16xf32> into f32
// CHECK:       %[[T7:.*]] = vector.from_elements %[[T3]], %[[T6]] : vector<2xf32>
gpu.module @xevm_module{
gpu.func @vector_multi_reduction_dim0_distributed_dim1_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
  %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2xf32>) {
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      : () -> (vector<32x16xf32>)
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [1]>}
      dense<0.0>  : vector<32xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_operand_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
        layout_operand_1 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [1]>,
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [1]>
      }
      [1] : vector<32x16xf32> to vector<32xf32>
    gpu.yield %1 : vector<32xf32>
  }
  "some_user_op"(%r) : (vector<2xf32>) -> ()
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction
// CHECK:     %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<2xf32>) {
// CHECK:       %[[SRC:.*]] = "some_def"() {{.*}} : () -> vector<16x2xf32>
// CHECK:       %[[T1:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:    {offsets = [0, 0], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK:       %[[T2:.*]] = vector.shape_cast %[[T1]] {{.*}} : vector<16x1xf32> to vector<16xf32>
// CHECK:       %[[T3:.*]] = vector.reduction <add>, %[[T2]], %{{.*}} : vector<16xf32> into f32
// CHECK:       %[[T4:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:     {offsets = [0, 1], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK:       %[[T5:.*]] = vector.shape_cast %[[T4]] {{.*}} : vector<16x1xf32> to vector<16xf32>
// CHECK:       %[[T6:.*]] = vector.reduction <add>, %[[T5]], %{{.*}} : vector<16xf32> into f32
// CHECK:       %[[T7:.*]] = vector.from_elements %[[T3]], %[[T6]] : vector<2xf32>
// CHECK:       gpu.yield %[[T7]] : vector<2xf32>
// CHECK:     }
gpu.module @xevm_module{
gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
  %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2xf32>) {
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      : () -> (vector<16x2xf32>)
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
      dense<0.0>  : vector<2xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_operand_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
        layout_operand_1 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>,
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>
      }
      [0] : vector<16x2xf32> to vector<2xf32>
    gpu.yield %1 : vector<2xf32>
  }
  "some_user_op"(%r) : (vector<2xf32>) -> ()
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @scatter_ops_chunksize({{.*}}) {
// CHECK:       %[[OFFSETS:.*]] = arith.constant {{.*}} dense<12> : vector<16xindex>
// CHECK:       %[[MASKS:.*]] = arith.constant {{.*}} dense<true> : vector<16xi1>
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%{{.*}})[16]
// CHECK-SAME:    -> (vector<1x8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>) {
// CHECK:         gpu.yield %{{.*}}, %{{.*}}, %[[OFFSETS]], %[[MASKS]] :
// CHECK-SAME:      vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = xegpu.load %[[W]]#1[%[[W]]#2], %[[W]]#3 <{chunk_size = 8 : i64}>
// CHECK-SAME:    : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
// CHECK-NEXT:  xegpu.store %[[T1]], %[[W]]#1[%[[W]]#2], %[[W]]#3 <{chunk_size = 8 : i64}>
// CHECK-SAME:    : vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.module @xevm_module{
  gpu.func @scatter_ops_chunksize(%laneid: index, %src: memref<256xf16>) {
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %1 = arith.constant
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        dense<1>: vector<16xi1>
      %offset = arith.constant
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        dense<12> : vector<16xindex>
      %3 = xegpu.load %src[%offset], %1 <{chunk_size=8}>
        {
          layout_operand_1 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
          layout_operand_2 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
        }
        : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
      xegpu.store %3, %src[%offset], %1 <{chunk_size=8}>
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>,
          layout_operand_2 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
          layout_operand_3 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
        }
        : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @scatter_ops({{.*}}) {
// CHECK:       %[[OFFSETS:.*]] = arith.constant {{.*}} dense<12> : vector<16xindex>
// CHECK:       %[[MASKS:.*]] = arith.constant {{.*}} dense<true> : vector<16xi1>
// CHECK:       %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%{{.*}})[16]
// CHECK-SAME:    -> (vector<1xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>) {
// CHECK:         gpu.yield %{{.*}}, %{{.*}}, %[[OFFSETS]], %[[MASKS]]
// CHECK-SAME:    : vector<16xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = xegpu.load %[[W]]#1[%[[W]]#2], %[[W]]#3
// CHECK-SAME:    : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<1xf16>
// CHECK-NEXT:  xegpu.store %[[T1]], %[[W]]#1[%[[W]]#2], %[[W]]#3
// CHECK-SAME:    : vector<1xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.module @xevm_module{
  gpu.func @scatter_ops(%src: memref<256xf16>, %laneid: index) {
    gpu.warp_execute_on_lane_0(%laneid)[16] {
      %1 = arith.constant
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        dense<1> : vector<16xi1>
      %offset = arith.constant
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
        dense<12> : vector<16xindex>
      %3 = xegpu.load %src[%offset], %1
      {
        layout_operand_1 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
        layout_operand_2 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
        layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
      } : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16xf16>
      xegpu.store %3, %src[%offset], %1
      {
        layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
        layout_operand_2 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
        layout_operand_3 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
      }
      : vector<16xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @memref_extract_aligned_pointer_as_index(
// CHECK:       %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (index, memref<256x256xf16>) {
// CHECK:         gpu.yield %{{.*}}, %{{.*}} : index, memref<256x256xf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[W]]#1 : memref<256x256xf16> -> index
// CHECK-NEXT:  arith.index_cast %[[INTPTR]] : index to i64
gpu.module @xevm_module{
  gpu.func @memref_extract_aligned_pointer_as_index(%arg0 : memref<256x256xf16>, %laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (index) {
      %ptr = memref.extract_aligned_pointer_as_index %arg0 : memref<256x256xf16> -> index
      gpu.yield %ptr : index
    }
    %ptr_i64 = arith.index_cast %r : index to i64
    "some_user_op"(%ptr_i64) : (i64) -> ()
    gpu.return
  }
}


// -----
// CHECK-LABEL: gpu.func @vector_transpose(
// CHECK:       %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<2x1xf32>, vector<1x2xf32>) {
// CHECK:         %[[SRC:.*]] = "some_op"() {{.*}} : () -> vector<16x2xf32>
// CHECK:         gpu.yield %{{.*}}, %[[SRC]] : vector<2x16xf32>, vector<16x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = vector.transpose %[[W]]#1, [1, 0] : vector<1x2xf32> to vector<2x1xf32>
gpu.module @xevm_module{
  gpu.func @vector_transpose(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<2x1xf32>) {
      %cst = "some_op"()
        {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
        : () -> (vector<16x2xf32>)
      %transpose = vector.transpose %cst, [1, 0]
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [16 , 1], lane_data = [1, 1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : vector<16x2xf32> to vector<2x16xf32>
      gpu.yield %transpose : vector<2x16xf32>
    }
    "some_user_op"(%r) : (vector<2x1xf32>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_bitcast(
// CHECK:       %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<4x1xi16>, vector<4x2xi8>) {
// CHECK:         %[[SRC:.*]] = "some_op"() {{.*}} : () -> vector<4x32xi8>
// CHECK:         gpu.yield %{{.*}}, %[[SRC]] : vector<4x16xi16>, vector<4x32xi8>
// CHECK:       }
// CHECK:       vector.bitcast %[[W]]#1 : vector<4x2xi8> to vector<4x1xi16>
gpu.module @xevm_module{
  gpu.func @vector_bitcast(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<4x1xi16>) {
      %cst = "some_op"()
        {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>}
        : () -> (vector<4x32xi8>)
      %bitcast = vector.bitcast %cst
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : vector<4x32xi8> to vector<4x16xi16>
      gpu.yield %bitcast : vector<4x16xi16>
    }
    "some_user_op"(%r) : (vector<4x1xi16>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_shapecast_rank_increasing
// CHECK:         %{{.*}}:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<1x1xf32>, vector<1xf32>) {
// CHECK:           gpu.yield %{{.*}} : vector<1x16xf32>, vector<16xf32>
// CHECK:         }
// CHECK:         %{{.*}} = vector.shape_cast %{{.*}}#1 : vector<1xf32> to vector<1x1xf32>
gpu.module @xevm_module {
  gpu.func @vector_shapecast_rank_increasing(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<1x1xf32>) {
      %cst = "some_op"()
        {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
        : () -> (vector<16xf32>)
      %cast = vector.shape_cast %cst
        {
          layout_operand_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : vector<16xf32> to vector<1x16xf32>
      gpu.yield %cast : vector<1x16xf32>
    }
    "some_user_op"(%r) : (vector<1x1xf32>) -> ()
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_shapecast_rank_reducing(
// CHECK:         %{{.*}}:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<1xf32>, vector<1x1xf32>) {
// CHECK:           gpu.yield %{{.*}} : vector<16xf32>, vector<1x16xf32>
// CHECK:         }
// CHECK:         %{{.*}} = vector.shape_cast %{{.*}}#1 : vector<1x1xf32> to vector<1xf32>
gpu.module @xevm_module {
  gpu.func @vector_shapecast_rank_reducing(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<1xf32>) {
      %cst = "some_op"()
        {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : () -> (vector<1x16xf32>)
      %cast = vector.shape_cast %cst
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>
        }
        : vector<1x16xf32> to vector<16xf32>
      gpu.yield %cast : vector<16xf32>
    }
    "some_user_op"(%r) : (vector<1xf32>) -> ()
    gpu.return
  }
}

// -----
// NOTE: Layouts are still valid, but distribution still requires a slice layout for the operand.
//
// CHECK-LABEL:  gpu.func @vector_shapecast_unsupported
// CHECK:          %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[16] -> (vector<1x1xf32>) {
// CHECK:            %[[T1:.*]] = vector.shape_cast %{{.*}} : vector<16xf32> to vector<1x16xf32>
// CHECK:            gpu.yield %[[T1]] : vector<1x16xf32>
// CHECK:          }
// CHECK:          "some_user_op"(%[[W]]) : (vector<1x1xf32>) -> ()
// CHECK:          gpu.return
gpu.module @xevm_module {
  gpu.func @vector_shapecast_unsupported(%laneid: index) {
    %r = gpu.warp_execute_on_lane_0(%laneid)[16] -> (vector<1x1xf32>) {
      %cst = "some_op"()
        {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]> }
        : () -> (vector<16xf32>)
      %cast = vector.shape_cast %cst
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : vector<16xf32> to vector<1x16xf32>
      gpu.yield %cast : vector<1x16xf32>
    }
    "some_user_op"(%r) : (vector<1x1xf32>) -> ()
    gpu.return
  }
}
