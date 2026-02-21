
// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=pvc' --allow-unregistered-dialect \
// RUN: --test-xegpu-sg-to-wi-distribute-experimental --split-input-file %s | FileCheck %s

gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @create_nd_tdesc
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TD:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>
gpu.func @create_nd_tdesc(%arg0: memref<256x256xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf16>
    -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @cerate_nd_tedesc_nonmemref_source
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TD:.*]] = xegpu.create_nd_tdesc %{{.*}}, shape : [256, 256], strides : [256, 1] : ui64 -> !xegpu.tensor_desc<16x16xf16>
gpu.func @cerate_nd_tedesc_nonmemref_source(%arg0: ui64) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0, shape : [256, 256], strides : [256, 1] : ui64
    -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @load_nd
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<16xf16> to vector<16x1xf16>
gpu.func @load_nd() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
  gpu.return
}

// CHECK-LABEL: gpu.func @load_nd_packed
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<16xf16> to vector<16x1xf16>
gpu.func @load_nd_packed() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  %1 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
  gpu.return
}

// CHECK-LABEL: gpu.func @load_nd_transpose
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x8xf32> -> vector<8xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<8xf32> to vector<1x8xf32>
gpu.func @load_nd_transpose() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x8xf32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
  %1 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x8xf32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @store_nd
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[CAST2:.*]] = vector.shape_cast %[[LOAD]] : vector<16xf16> to vector<16x1xf16>
// CHECK: %[[CAST3:.*]] = vector.shape_cast %[[CAST2]] : vector<16x1xf16> to vector<16xf16>
// CHECK: xegpu.store_nd %[[CAST3]], %{{.*}}[%[[C0]], %[[C0]]] : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.func @store_nd() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %2 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
  xegpu.store_nd %2, %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @dpas
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x1xf32>
// CHECK-DAG: %[[LOAD0:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK-DAG: %[[CAST2:.*]] = vector.shape_cast %[[LOAD0]] : vector<8xf16> to vector<8x1xf16>
// CHECK-DAG: %[[LOAD1:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK-DAG: %[[CAST3:.*]] = vector.shape_cast %[[LOAD1]] : vector<16xf16> to vector<16x1xf16>
// CHECK-DAG: %[[CAST4:.*]] = vector.shape_cast %[[CST]] : vector<8x1xf32> to vector<8xf32>
// CHECK-DAG: %[[CAST5:.*]] = vector.shape_cast %[[CAST3]] : vector<16x1xf16> to vector<16xf16>
// CHECK-DAG: %[[CAST6:.*]] = vector.shape_cast %[[CAST2]] : vector<8x1xf16> to vector<8xf16>
// CHECK: %[[DPAS:.*]] = xegpu.dpas %[[CAST6]], %[[CAST5]], %[[CAST4]] : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
// CHECK: %[[CAST7:.*]] = vector.shape_cast %[[DPAS]] : vector<8xf32> to vector<8x1xf32>
// CHECK: gpu.return
gpu.func @dpas() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  %5 = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    dense<0.0> : vector<8x16xf32>
  %2 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %5
    {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
     layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
     layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>  -> vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @elementwise
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16x1xf32>
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %{{.*}}[%[[C0]], %[[C0]]] : !xegpu.tensor_desc<16x16xf32> -> vector<16xf32>
// CHECK: %[[CAST2:.*]] = vector.shape_cast %[[LOAD]] : vector<16xf32> to vector<16x1xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[CAST2]], %[[CST]] : vector<16x1xf32>
// CHECK: gpu.return
gpu.func @elementwise() {
  %c0 = arith.constant 0 : index
  %0 = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    dense<1.0> : vector<16x16xf32>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %2 = xegpu.load_nd %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf32>
  %3 = arith.addf %0, %2
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @arith_constant
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16x1xf32>
// CHECK: gpu.return
gpu.func @arith_constant() {
  %0 = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    dense<1.0> : vector<16x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @prefetch_nd
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: xegpu.prefetch_nd %{{.*}}[%[[C0]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16>
// CHECK: gpu.return
gpu.func @prefetch_nd() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  xegpu.prefetch_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_reduction
// CHECK:     %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:     %[[LANE_RED:.*]] = vector.reduction <add>, %[[CAST:.*]] : vector<2xf32> into f32
// CHECK-DAG: %[[C16_1:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK:     %[[SHUFFLE1:.*]], %{{.*}} = gpu.shuffle  xor %[[LANE_RED]], %[[C1]], %[[C16_1]] : f32
// CHECK:     %[[ADD1:.*]] = arith.addf %[[LANE_RED]], %[[SHUFFLE1]] : f32
// CHECK-DAG: %[[C16_2:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK:     %[[SHUFFLE2:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD1]], %[[C2]], %[[C16_2]] : f32
// CHECK:     %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[SHUFFLE2]] : f32
// CHECK-DAG: %[[C16_3:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK:     %[[SHUFFLE3:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD2]], %[[C4]], %[[C16_3]] : f32
// CHECK:     %[[ADD3:.*]] = arith.addf %[[ADD2]], %[[SHUFFLE3]] : f32
// CHECK-DAG: %[[C16_4:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK:     %[[SHUFFLE4:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD3]], %[[C8]], %[[C16_4]] : f32
// CHECK:     %[[ADD4:.*]] = arith.addf %[[ADD3]], %[[SHUFFLE4]] : f32
// CHECK:     %[[FINAL:.*]] = arith.addf %[[ADD4]], %[[CST]] : f32
gpu.func @vector_reduction() {
  %acc = arith.constant 1.0 : f32
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} : () -> vector<32xf32>
  %2 = vector.reduction <add>, %0, %acc : vector<32xf32> into f32
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction
// CHECK: %0 = vector.extract_strided_slice %cst {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x1xf32> to vector<1x1xf32>
// CHECK: %1 = vector.shape_cast %0 : vector<1x1xf32> to vector<1xf32>
// CHECK: %2 = vector.extract %cst_0[0] : f32 from vector<2xf32>
// CHECK: %3 = vector.reduction <add>, %1 : vector<1xf32> into f32
// CHECK: %c16_i32 = arith.constant 16 : i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %shuffleResult, %valid = gpu.shuffle  xor %3, %c1_i32, %c16_i32 : f32
// CHECK: %4 = arith.addf %3, %shuffleResult : f32
// CHECK: %c16_i32_2 = arith.constant 16 : i32
// CHECK: %c2_i32 = arith.constant 2 : i32
// CHECK: %shuffleResult_3, %valid_4 = gpu.shuffle  xor %4, %c2_i32, %c16_i32_2 : f32
// CHECK: %5 = arith.addf %4, %shuffleResult_3 : f32
// CHECK: %c16_i32_5 = arith.constant 16 : i32
// CHECK: %c4_i32 = arith.constant 4 : i32
// CHECK: %shuffleResult_6, %valid_7 = gpu.shuffle  xor %5, %c4_i32, %c16_i32_5 : f32
// CHECK: %6 = arith.addf %5, %shuffleResult_6 : f32
// CHECK: %c16_i32_8 = arith.constant 16 : i32
// CHECK: %c8_i32 = arith.constant 8 : i32
// CHECK: %shuffleResult_9, %valid_10 = gpu.shuffle  xor %6, %c8_i32, %c16_i32_8 : f32
// CHECK: %7 = arith.addf %6, %shuffleResult_9 : f32
// CHECK: %8 = arith.addf %7, %2 : f32
// CHECK: %9 = vector.insert %8, %cst_1 [0] : f32 into vector<2xf32>
// CHECK: %10 = vector.extract_strided_slice %cst {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x1xf32> to vector<1x1xf32>
// CHECK: %11 = vector.shape_cast %10 : vector<1x1xf32> to vector<1xf32>
// CHECK: %12 = vector.extract %cst_0[1] : f32 from vector<2xf32>
// CHECK: %13 = vector.reduction <add>, %11 : vector<1xf32> into f32
// CHECK: %c16_i32_11 = arith.constant 16 : i32
// CHECK: %c1_i32_12 = arith.constant 1 : i32
// CHECK: %shuffleResult_13, %valid_14 = gpu.shuffle  xor %13, %c1_i32_12, %c16_i32_11 : f32
// CHECK: %14 = arith.addf %13, %shuffleResult_13 : f32
// CHECK: %c16_i32_15 = arith.constant 16 : i32
// CHECK: %c2_i32_16 = arith.constant 2 : i32
// CHECK: %shuffleResult_17, %valid_18 = gpu.shuffle  xor %14, %c2_i32_16, %c16_i32_15 : f32
// CHECK: %15 = arith.addf %14, %shuffleResult_17 : f32
// CHECK: %c16_i32_19 = arith.constant 16 : i32
// CHECK: %c4_i32_20 = arith.constant 4 : i32
// CHECK: %shuffleResult_21, %valid_22 = gpu.shuffle  xor %15, %c4_i32_20, %c16_i32_19 : f32
// CHECK: %16 = arith.addf %15, %shuffleResult_21 : f32
// CHECK: %c16_i32_23 = arith.constant 16 : i32
// CHECK: %c8_i32_24 = arith.constant 8 : i32
// CHECK: %shuffleResult_25, %valid_26 = gpu.shuffle  xor %16, %c8_i32_24, %c16_i32_23 : f32
// CHECK: %17 = arith.addf %16, %shuffleResult_25 : f32
// CHECK: %18 = arith.addf %17, %12 : f32
// CHECK: %19 = vector.insert %18, %9 [1] : f32 into vector<2xf32>
gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
  %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      dense<0.0>  : vector<2x16xf32>
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
      dense<0.0>  : vector<2xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>
      }
      [1] : vector<2x16xf32> to vector<2xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction
// CHECK: %2 = vector.extract_strided_slice %1 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x2xf32> to vector<1x1xf32>
// CHECK: %3 = vector.shape_cast %2 : vector<1x1xf32> to vector<1xf32>
// CHECK: %4 = vector.extract %cst[0] : f32 from vector<2xf32>
// CHECK: %5 = vector.reduction <add>, %3 : vector<1xf32> into f32
// CHECK: %c16_i32 = arith.constant 16 : i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %shuffleResult, %valid = gpu.shuffle  xor %5, %c1_i32, %c16_i32 : f32
// CHECK: %6 = arith.addf %5, %shuffleResult : f32
// CHECK: %c16_i32_1 = arith.constant 16 : i32
// CHECK: %c2_i32 = arith.constant 2 : i32
// CHECK: %shuffleResult_2, %valid_3 = gpu.shuffle  xor %6, %c2_i32, %c16_i32_1 : f32
// CHECK: %7 = arith.addf %6, %shuffleResult_2 : f32
// CHECK: %c16_i32_4 = arith.constant 16 : i32
// CHECK: %c4_i32 = arith.constant 4 : i32
// CHECK: %shuffleResult_5, %valid_6 = gpu.shuffle  xor %7, %c4_i32, %c16_i32_4 : f32
// CHECK: %8 = arith.addf %7, %shuffleResult_5 : f32
// CHECK: %c16_i32_7 = arith.constant 16 : i32
// CHECK: %c8_i32 = arith.constant 8 : i32
// CHECK: %shuffleResult_8, %valid_9 = gpu.shuffle  xor %8, %c8_i32, %c16_i32_7 : f32
// CHECK: %9 = arith.addf %8, %shuffleResult_8 : f32
// CHECK: %10 = arith.addf %9, %4 : f32
// CHECK: %11 = vector.insert %10, %cst_0 [0] : f32 into vector<2xf32>
// CHECK: %12 = vector.extract_strided_slice %1 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x2xf32> to vector<1x1xf32>
// CHECK: %13 = vector.shape_cast %12 : vector<1x1xf32> to vector<1xf32>
// CHECK: %14 = vector.extract %cst[1] : f32 from vector<2xf32>
// CHECK: %15 = vector.reduction <add>, %13 : vector<1xf32> into f32
// CHECK: %c16_i32_10 = arith.constant 16 : i32
// CHECK: %c1_i32_11 = arith.constant 1 : i32
// CHECK: %shuffleResult_12, %valid_13 = gpu.shuffle  xor %15, %c1_i32_11, %c16_i32_10 : f32
// CHECK: %16 = arith.addf %15, %shuffleResult_12 : f32
// CHECK: %c16_i32_14 = arith.constant 16 : i32
// CHECK: %c2_i32_15 = arith.constant 2 : i32
// CHECK: %shuffleResult_16, %valid_17 = gpu.shuffle  xor %16, %c2_i32_15, %c16_i32_14 : f32
// CHECK: %17 = arith.addf %16, %shuffleResult_16 : f32
// CHECK: %c16_i32_18 = arith.constant 16 : i32
// CHECK: %c4_i32_19 = arith.constant 4 : i32
// CHECK: %shuffleResult_20, %valid_21 = gpu.shuffle  xor %17, %c4_i32_19, %c16_i32_18 : f32
// CHECK: %18 = arith.addf %17, %shuffleResult_20 : f32
// CHECK: %c16_i32_22 = arith.constant 16 : i32
// CHECK: %c8_i32_23 = arith.constant 8 : i32
// CHECK: %shuffleResult_24, %valid_25 = gpu.shuffle  xor %18, %c8_i32_23, %c16_i32_22 : f32
// CHECK: %19 = arith.addf %18, %shuffleResult_24 : f32
// CHECK: %20 = arith.addf %19, %14 : f32
// CHECK: %21 = vector.insert %20, %11 [1] : f32 into vector<2xf32>
gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      : () -> (vector<16x2xf32>)
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
      dense<0.0>  : vector<2xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>
      }
      [0] : vector<16x2xf32> to vector<2xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim0_reduction
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4x1xf32>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK:         %[[RED:.*]] = vector.multi_reduction <add>, %[[CST]], %[[CST_0]] [0] : vector<4x1xf32> to vector<1xf32>
// CHECK:         gpu.return
gpu.func @vector_multi_reduction_dim1_distributed_dim0_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
    %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      dense<0.0>  : vector<4x16xf32>
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
      dense<0.0>  : vector<16xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>
      }
      [0] : vector<4x16xf32> to vector<16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_dim0_distributed_dim1_reduction
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x12xf32>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK:         %[[RED:.*]] = vector.multi_reduction <add>, %[[CST]], %[[CST_0]] [1] : vector<1x12xf32> to vector<1xf32>
// CHECK:         gpu.return
gpu.func @vector_multi_reduction_dim0_distributed_dim1_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
    %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      dense<0.0>  : vector<16x12xf32>
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [1]>}
      dense<0.0>  : vector<16xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [1]>
      }
      [1] : vector<16x12xf32> to vector<16xf32>
  gpu.return
}
}
