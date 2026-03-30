
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

// CHECK-LABEL: gpu.func @scatter_load_chunksize
// CHECK: %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME: : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<8xf16> to vector<1x8xf16>
gpu.func @scatter_load_chunksize(%src: memref<256xf16>) {
  %offset = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<12> : vector<16xindex>
  %mask = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<true> : vector<16xi1>
  %0 = xegpu.load %src[%offset], %mask
    <{chunk_size = 8, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
    : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
  gpu.return
}

// CHECK-LABEL: gpu.func @scatter_store_chunksize
// CHECK: %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME: : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
// CHECK: %[[C1:.*]] = vector.shape_cast %[[LOAD]] : vector<8xf16> to vector<1x8xf16>
// CHECK: %[[C2:.*]] = vector.shape_cast %[[C1]] : vector<1x8xf16> to vector<8xf16>
// CHECK: xegpu.store %[[C2]], %arg0[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME: : vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.func @scatter_store_chunksize(%src: memref<256xf16>) {
  %offset = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<12> : vector<16xindex>
  %mask = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<true> : vector<16xi1>
  %0 = xegpu.load %src[%offset], %mask
    <{chunk_size = 8, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
    : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
  xegpu.store %0, %src[%offset], %mask
    <{chunk_size = 8, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}>
    : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
  gpu.return
}

// CHECK-LABEL: gpu.func @scatter_load
// CHECK: %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[OFFSET]]], %[[MASK]]
// CHECK-SAME: : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<1xf16>
gpu.func @scatter_load(%src: memref<256xf16>) {
  %offset = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<12> : vector<16xindex>
  %mask = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<true> : vector<16xi1>
  %0 = xegpu.load %src[%offset], %mask
    <{layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>}>
    : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16xf16>
  gpu.return
}

// CHECK-LABEL: gpu.func @scatter_store
// CHECK: %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[OFFSET]]], %[[MASK]]
// CHECK-SAME: : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<1xf16>
// CHECK: xegpu.store %[[LOAD]], %arg0[%[[OFFSET]]], %[[MASK]]
// CHECK-SAME: : vector<1xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.func @scatter_store(%src: memref<256xf16>) {
  %offset = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<12> : vector<16xindex>
  %mask = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    dense<true> : vector<16xi1>
  %0 = xegpu.load %src[%offset], %mask
    <{layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>}>
    : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16xf16>
  xegpu.store %0, %src[%offset], %mask
    <{layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>}>
    : vector<16xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
  gpu.return
}

// CHECK-LABEL: gpu.func @scatter_ops_with_leading_dims
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<1x1x1xi1>
// CHECK: %[[OFFSET:.*]] = arith.constant dense<12> : vector<1x1x1xindex>
// CHECK: %[[V1:.*]] = vector.shape_cast %[[OFFSET]] : vector<1x1x1xindex> to vector<1xindex>
// CHECK: %[[V2:.*]] = vector.shape_cast %[[MASK]] : vector<1x1x1xi1> to vector<1xi1>
// CHECK: %[[LOAD:.*]] = xegpu.load %arg0[%[[V1]]], %[[V2]]
// CHECK-SAME: : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<1xf16>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[LOAD]] : vector<1xf16> to vector<1x1x1xf16>
// CHECK: %[[CAST2:.*]] = vector.shape_cast %[[CAST]] : vector<1x1x1xf16> to vector<1xf16>
// CHECK: %[[V3:.*]] = vector.shape_cast %[[OFFSET]] : vector<1x1x1xindex> to vector<1xindex>
// CHECK: %[[V4:.*]] = vector.shape_cast %[[MASK]] : vector<1x1x1xi1> to vector<1xi1>
// CHECK: xegpu.store %[[CAST2]], %arg0[%[[V3]]], %[[V4]]
// CHECK-SAME: : vector<1xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.func @scatter_ops_with_leading_dims(%src: memref<256xf16>) {
  %mask = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>}
    dense<1> : vector<1x1x16xi1>
  %offset = arith.constant
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>}
    dense<12> : vector<1x1x16xindex>
  %0 = xegpu.load %src[%offset], %mask
    <{layout = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>}>
    : memref<256xf16>, vector<1x1x16xindex>, vector<1x1x16xi1> -> vector<1x1x16xf16>
  xegpu.store %0, %src[%offset], %mask
    <{layout = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>}>
    : vector<1x1x16xf16>, memref<256xf16>, vector<1x1x16xindex>, vector<1x1x16xi1>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_reduction
// CHECK:     %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:     %[[LANE_RED:.*]] = vector.reduction <add>, %[[CAST:.*]] : vector<2xf32> into f32
// CHECK-DAG: %[[C16_1:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK:     %[[SHUFFLE1:.*]], %{{.*}} = gpu.shuffle xor %[[LANE_RED]], %[[C1]], %[[C16_1]] : f32
// CHECK:     %[[ADD1:.*]] = arith.addf %[[LANE_RED]], %[[SHUFFLE1]] : f32
// CHECK-DAG: %[[C16_2:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK:     %[[SHUFFLE2:.*]], %{{.*}} = gpu.shuffle xor %[[ADD1]], %[[C2]], %[[C16_2]] : f32
// CHECK:     %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[SHUFFLE2]] : f32
// CHECK-DAG: %[[C16_3:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK:     %[[SHUFFLE3:.*]], %{{.*}} = gpu.shuffle xor %[[ADD2]], %[[C4]], %[[C16_3]] : f32
// CHECK:     %[[ADD3:.*]] = arith.addf %[[ADD2]], %[[SHUFFLE3]] : f32
// CHECK-DAG: %[[C16_4:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK:     %[[SHUFFLE4:.*]], %{{.*}} = gpu.shuffle xor %[[ADD3]], %[[C8]], %[[C16_4]] : f32
// CHECK:     %[[ADD4:.*]] = arith.addf %[[ADD3]], %[[SHUFFLE4]] : f32
// CHECK:     %[[FINAL:.*]] = arith.addf %[[ADD4]], %[[CST]] : f32
gpu.func @vector_reduction() {
  %acc = arith.constant 1.0 : f32
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} : () -> vector<32xf32>
  %2 = vector.reduction <add>, %0, %acc : vector<32xf32> into f32
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x1xf32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK: %[[V0:.*]] = vector.extract_strided_slice %[[CST]] {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x1xf32> to vector<1x1xf32>
// CHECK: %[[V1:.*]] = vector.shape_cast %[[V0]] : vector<1x1xf32> to vector<1xf32>
// CHECK: %[[V2:.*]] = vector.extract %[[CST_0]][0] : f32 from vector<2xf32>
// CHECK: %[[V3:.*]] = vector.reduction <add>, %[[V1]] : vector<1xf32> into f32
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[SHUFFLE1:.*]], %{{.*}} = gpu.shuffle xor %[[V3]], %[[C1]], %[[C16]] : f32
// CHECK: %[[V4:.*]] = arith.addf %[[V3]], %[[SHUFFLE1]] : f32
// CHECK-DAG: %[[C16_2:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[SHUFFLE2:.*]], %{{.*}} = gpu.shuffle xor %[[V4]], %[[C2]], %[[C16_2]] : f32
// CHECK: %[[V5:.*]] = arith.addf %[[V4]], %[[SHUFFLE2]] : f32
// CHECK-DAG: %[[C16_3:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[SHUFFLE3:.*]], %{{.*}} = gpu.shuffle xor %[[V5]], %[[C4]], %[[C16_3]] : f32
// CHECK: %[[V6:.*]] = arith.addf %[[V5]], %[[SHUFFLE3]] : f32
// CHECK-DAG: %[[C16_4:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[SHUFFLE4:.*]], %{{.*}} = gpu.shuffle xor %[[V6]], %[[C8]], %[[C16_4]] : f32
// CHECK: %[[V7:.*]] = arith.addf %[[V6]], %[[SHUFFLE4]] : f32
// CHECK: %[[V8:.*]] = arith.addf %[[V7]], %[[V2]] : f32
// CHECK: %[[V9:.*]] = vector.insert %[[V8]], %[[CST_1]] [0] : f32 into vector<2xf32>
// CHECK: %[[V10:.*]] = vector.extract_strided_slice %[[CST]] {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x1xf32> to vector<1x1xf32>
// CHECK: %[[V11:.*]] = vector.shape_cast %[[V10]] : vector<1x1xf32> to vector<1xf32>
// CHECK: %[[V12:.*]] = vector.extract %[[CST_0]][1] : f32 from vector<2xf32>
// CHECK: %[[V13:.*]] = vector.reduction <add>, %[[V11]] : vector<1xf32> into f32
// CHECK-DAG: %[[C16_5:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1_2:.*]] = arith.constant 1 : i32
// CHECK: %[[SHUFFLE5:.*]], %{{.*}} = gpu.shuffle xor %[[V13]], %[[C1_2]], %[[C16_5]] : f32
// CHECK: %[[V14:.*]] = arith.addf %[[V13]], %[[SHUFFLE5]] : f32
// CHECK-DAG: %[[C16_6:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2_2:.*]] = arith.constant 2 : i32
// CHECK: %[[SHUFFLE6:.*]], %{{.*}} = gpu.shuffle xor %[[V14]], %[[C2_2]], %[[C16_6]] : f32
// CHECK: %[[V15:.*]] = arith.addf %[[V14]], %[[SHUFFLE6]] : f32
// CHECK-DAG: %[[C16_7:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4_2:.*]] = arith.constant 4 : i32
// CHECK: %[[SHUFFLE7:.*]], %{{.*}} = gpu.shuffle xor %[[V15]], %[[C4_2]], %[[C16_7]] : f32
// CHECK: %[[V16:.*]] = arith.addf %[[V15]], %[[SHUFFLE7]] : f32
// CHECK-DAG: %[[C16_8:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8_2:.*]] = arith.constant 8 : i32
// CHECK: %[[SHUFFLE8:.*]], %{{.*}} = gpu.shuffle xor %[[V16]], %[[C8_2]], %[[C16_8]] : f32
// CHECK: %[[V17:.*]] = arith.addf %[[V16]], %[[SHUFFLE8]] : f32
// CHECK: %[[V18:.*]] = arith.addf %[[V17]], %[[V12]] : f32
// CHECK: %[[V19:.*]] = vector.insert %[[V18]], %[[V9]] [1] : f32 into vector<2xf32>
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
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x2xf32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK: %[[V0:.*]] = vector.extract_strided_slice %[[CST]] {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x2xf32> to vector<1x1xf32>
// CHECK: %[[V1:.*]] = vector.shape_cast %[[V0]] : vector<1x1xf32> to vector<1xf32>
// CHECK: %[[V2:.*]] = vector.extract %[[CST_0]][0] : f32 from vector<2xf32>
// CHECK: %[[V3:.*]] = vector.reduction <add>, %[[V1]] : vector<1xf32> into f32
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[SHUFFLE1:.*]], %{{.*}} = gpu.shuffle xor %[[V3]], %[[C1]], %[[C16]] : f32
// CHECK: %[[V4:.*]] = arith.addf %[[V3]], %[[SHUFFLE1:.*]] : f32
// CHECK-DAG: %[[C16_2:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[SHUFFLE2:.*]], %{{.*}} = gpu.shuffle xor %[[V4]], %[[C2]], %[[C16_2]] : f32
// CHECK: %[[V5:.*]] = arith.addf %[[V4]], %[[SHUFFLE2]] : f32
// CHECK-DAG: %[[C16_3:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[SHUFFLE3:.*]], %{{.*}} = gpu.shuffle xor %[[V5]], %[[C4]], %[[C16_3]] : f32
// CHECK: %[[V6:.*]] = arith.addf %[[V5]], %[[SHUFFLE3]] : f32
// CHECK-DAG: %[[C16_4:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[SHUFFLE4:.*]], %{{.*}} = gpu.shuffle xor %[[V6]], %[[C8]], %[[C16_4]] : f32
// CHECK: %[[V7:.*]] = arith.addf %[[V6]], %[[SHUFFLE4]] : f32
// CHECK: %[[V8:.*]] = arith.addf %[[V7]], %[[V2]] : f32
// CHECK: %[[V9:.*]] = vector.insert %[[V8]], %[[CST_1]] [0] : f32 into vector<2xf32>
// CHECK: %[[V10:.*]] = vector.extract_strided_slice %[[CST]] {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x2xf32> to vector<1x1xf32>
// CHECK: %[[V11:.*]] = vector.shape_cast %[[V10]] : vector<1x1xf32> to vector<1xf32>
// CHECK: %[[V12:.*]] = vector.extract %[[CST_0]][1] : f32 from vector<2xf32>
// CHECK: %[[V13:.*]] = vector.reduction <add>, %[[V11]] : vector<1xf32> into f32
// CHECK-DAG: %[[C16_5:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C1_2:.*]] = arith.constant 1 : i32
// CHECK: %[[SHUFFLE5:.*]], %{{.*}} = gpu.shuffle xor %[[V13]], %[[C1_2]], %[[C16_5]] : f32
// CHECK: %[[V14:.*]] = arith.addf %[[V13]], %[[SHUFFLE5]] : f32
// CHECK-DAG: %[[C16_6:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C2_2:.*]] = arith.constant 2 : i32
// CHECK: %[[SHUFFLE6:.*]], %{{.*}} = gpu.shuffle xor %[[V14]], %[[C2_2]], %[[C16_6]] : f32
// CHECK: %[[V15:.*]] = arith.addf %[[V14]], %[[SHUFFLE6]] : f32
// CHECK-DAG: %[[C16_7:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C4_2:.*]] = arith.constant 4 : i32
// CHECK: %[[SHUFFLE7:.*]], %{{.*}} = gpu.shuffle xor %[[V15]], %[[C4_2]], %[[C16_7]] : f32
// CHECK: %[[V16:.*]] = arith.addf %[[V15]], %[[SHUFFLE7]] : f32
// CHECK-DAG: %[[C16_8:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C8_2:.*]] = arith.constant 8 : i32
// CHECK: %[[SHUFFLE8:.*]], %{{.*}} = gpu.shuffle xor %[[V16]], %[[C8_2]], %[[C16_8]] : f32
// CHECK: %[[V17:.*]] = arith.addf %[[V16]], %[[SHUFFLE8]] : f32
// CHECK: %[[V18:.*]] = arith.addf %[[V17]], %[[V12]] : f32
// CHECK: %[[V19:.*]] = vector.insert %[[V18]], %[[V9]] [1] : f32 into vector<2xf32>
gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
    %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      dense<0.0> : vector<16x2xf32>
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

// CHECK-LABEL: gpu.func @vector_multi_reduction_3d_leading_unit_dim_lane_local
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x16x2xf32>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<1x2xf32>
// CHECK:         %[[RED:.*]] = vector.multi_reduction <add>, %[[CST]], %[[CST_0]] [1] : vector<1x16x2xf32> to vector<1x2xf32>
// CHECK:         gpu.return
gpu.func @vector_multi_reduction_3d_leading_unit_dim_lane_local() {
    %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>}
      dense<0.0>  : vector<1x16x32xf32>
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [1]>}
      dense<0.0>  : vector<1x32xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [1]>
      }
      [1] : vector<1x16x32xf32> to vector<1x32xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_multi_reduction_3d_leading_unit_dim_cross_lane
// CHECK-DAG:     %[[SRC:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x2xf32>
// CHECK-DAG:     %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<1x2xf32>
// CHECK:         vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:      {offsets = [0, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1]}
// CHECK:         %[[ACC0:.*]] = vector.extract %[[ACC]][0, 0] : f32 from vector<1x2xf32>
// CHECK:         vector.reduction <add>, %{{.*}} : vector<1xf32> into f32
// CHECK-COUNT-4: gpu.shuffle xor %{{.*}} : f32
// CHECK:         %[[WITH_ACC0:.*]] = arith.addf %{{.*}}, %[[ACC0]] : f32
// CHECK:         %[[INS0:.*]] = vector.insert %[[WITH_ACC0]], %{{.*}} [0, 0] : f32 into vector<1x2xf32>
// CHECK:         vector.extract_strided_slice %[[SRC]]
// CHECK-SAME:      {offsets = [0, 0, 1], sizes = [1, 1, 1], strides = [1, 1, 1]}
// CHECK:         %[[ACC1:.*]] = vector.extract %[[ACC]][0, 1] : f32 from vector<1x2xf32>
// CHECK:         vector.reduction <add>, %{{.*}} : vector<1xf32> into f32
// CHECK-COUNT-4: gpu.shuffle xor %{{.*}} : f32
// CHECK:         %[[WITH_ACC1:.*]] = arith.addf %{{.*}}, %[[ACC1]] : f32
// CHECK:         vector.insert %[[WITH_ACC1]], %[[INS0]] [0, 1] : f32 into vector<1x2xf32>
// CHECK:         gpu.return
gpu.func @vector_multi_reduction_3d_leading_unit_dim_cross_lane() {
    %src = arith.constant
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16, 1], lane_data = [1, 1, 1]>}
      dense<0.0>  : vector<1x16x2xf32>
    %acc = arith.constant
      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16, 1], lane_data = [1, 1, 1]>, dims = [1]>}
      dense<0.0>  : vector<1x2xf32>
    %1 = vector.multi_reduction <add>, %src, %acc
      {
        layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16, 1], lane_data = [1, 1, 1]>, dims = [1]>
      }
      [1] : vector<1x16x2xf32> to vector<1x2xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_from_2d
// CHECK: %[[EXT:.*]] = vector.extract %{{.*}}[0] : vector<1xf32> from vector<4x1xf32>
gpu.func @vector_extract_from_2d() {
  %src = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<4x16xf32>
  %0 = vector.extract %src[0]
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : vector<16xf32> from vector<4x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_from_2d_offset2
// CHECK: %[[EXT:.*]] = vector.extract %{{.*}}[2] : vector<1xf32> from vector<8x1xf32>
gpu.func @vector_extract_from_2d_offset2() {
  %src = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<8x16xf32>
  %0 = vector.extract %src[2]
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : vector<16xf32> from vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_into_2d
// CHECK: %[[INS:.*]] = vector.insert %{{.*}}, %{{.*}}[0] : vector<1xf32> into vector<4x1xf32>
gpu.func @vector_insert_into_2d() {
  %val = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<16xf32>
  %dst = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<4x16xf32>
  %0 = vector.insert %val, %dst[0]
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16xf32> into vector<4x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_into_2d_offset2
// CHECK: %[[INS:.*]] = vector.insert %{{.*}}, %{{.*}}[2] : vector<1xf32> into vector<8x1xf32>
gpu.func @vector_insert_into_2d_offset2() {
  %val = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<16xf32>
  %dst = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<8x16xf32>
  %0 = vector.insert %val, %dst[2]
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16xf32> into vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_strided_slice_distributed_dim_fully_extracted
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<24x1xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_distributed_dim_fully_extracted() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8, 0], sizes = [8, 16], strides = [1, 1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x16xf32> to vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_strided_slice_inner_distributed
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 3], sizes = [8, 1], strides = [1, 1]} : vector<24x4xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_inner_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x64xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8, 48], sizes = [8, 16], strides = [1, 1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x64xf32> to vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_strided_slice_outer_distributed
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<2x16xf32> to vector<1x16xf32>
gpu.func @vector_extract_strided_slice_outer_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
    : () -> vector<32x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [16], sizes = [16], strides = [1],
      layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
    }
    : vector<32x16xf32> to vector<16x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_strided_slice_1d
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
gpu.func @vector_extract_strided_slice_1d() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<64xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [16], sizes = [32], strides = [1],
      layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
    }
    : vector<64xf32> to vector<32xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_extract_strided_slice_partial_offsets
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<24x1xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_partial_offsets() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8], sizes = [8], strides = [1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x16xf32> to vector<8x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_strided_slice_distributed_dim_fully_inserted
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [24, 0], strides = [1, 1]} : vector<16x1xf32> into vector<64x1xf32>
gpu.func @vector_insert_strided_slice_distributed_dim_fully_inserted() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<16x16xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<64x16xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [24, 0], strides = [1, 1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<64x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_strided_slice_inner_distributed
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [24, 1], strides = [1, 1]} : vector<16x1xf32> into vector<64x2xf32>
gpu.func @vector_insert_strided_slice_inner_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<16x16xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<64x32xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [24, 16], strides = [1, 1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<64x32xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_strided_slice_outer_distributed
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<1x16xf32> into vector<3x32xf32>
gpu.func @vector_insert_strided_slice_outer_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
    : () -> vector<16x16xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
    : () -> vector<48x32xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [32, 4], strides = [1, 1],
      layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<48x32xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_strided_slice_1d
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [1], strides = [1]} : vector<1xf32> into vector<3xf32>
gpu.func @vector_insert_strided_slice_1d() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<16xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<48xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [16], strides = [1],
      layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
    }
    : vector<16xf32> into vector<48xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @vector_insert_strided_slice_different_ranks
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [13, 0], strides = [1]} : vector<1xf32> into vector<64x1xf32>
gpu.func @vector_insert_strided_slice_different_ranks() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<16xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<64x16xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [13, 0], strides = [1],
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16xf32> into vector<64x16xf32>
  gpu.return
}

// CHECK-LABEL: gpu.func @convert_layout_removed_when_compatible
// CHECK-NOT: xegpu.convert_layout
gpu.func @convert_layout_removed_when_compatible() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<16xf32>
  %2 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1], lane_data = [1]>}
    : () -> vector<1xf32>
  %1 = xegpu.convert_layout %0
    <{input_layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
    target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}>
    : vector<16xf32>
  %3 = xegpu.convert_layout %2
    <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>, 
    target_layout = #xegpu.layout<lane_layout = [1], lane_data = [1]>}> 
    : vector<1xf32>
  %4 = xegpu.convert_layout %3
    <{input_layout = #xegpu.layout<lane_layout = [1], lane_data = [1]>,
    target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [0, 1]>}> 
    : vector<1xf32>
  gpu.return
}
}

// -----
// load_matrix and store_matrix with coordinate computation (offsets [0,0])
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @load_store_matrix_1
// CHECK-DAG: %[[LANE_ID1:.*]] = gpu.lane_id
// CHECK-DAG: %[[R1:.*]] = arith.remui %[[LANE_ID1]], %{{.*}} : index
// CHECK-DAG: %[[D1:.*]] = arith.divui %[[LANE_ID1]], %{{.*}} : index
// CHECK-DAG: %[[R2:.*]] = arith.remui %[[D1]], %{{.*}} : index
// CHECK-DAG: %[[ROW:.*]] = arith.remui %[[R2]], %{{.*}} : index
// CHECK-DAG: %[[COL:.*]] = arith.remui %[[R1]], %{{.*}} : index
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%[[ROW]], %[[COL]]] : !xegpu.mem_desc<32x32xf32>, index, index -> vector<1x1xf32>
// CHECK: %[[LANE_ID2:.*]] = gpu.lane_id
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%{{.*}}, %{{.*}}] : vector<1x1xf32>, !xegpu.mem_desc<32x32xf32>, index, index
gpu.func @load_store_matrix_1(%arg0: !xegpu.mem_desc<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %1 = xegpu.load_matrix %arg0[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}> : !xegpu.mem_desc<32x32xf32>, index, index -> vector<2x8xf32>
  xegpu.store_matrix %1, %arg0[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}> : vector<2x8xf32>, !xegpu.mem_desc<32x32xf32>, index, index
  gpu.return
}
}

// -----
// load_matrix and store_matrix with non-zero offsets [0,1]
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @load_store_matrix_2
// CHECK-DAG: %[[LANE_ID1:.*]] = gpu.lane_id
// CHECK-DAG: %[[R1:.*]] = arith.remui %[[LANE_ID1]], %{{.*}} : index
// CHECK-DAG: %[[D1:.*]] = arith.divui %[[LANE_ID1]], %{{.*}} : index
// CHECK-DAG: %[[R2:.*]] = arith.remui %[[D1]], %{{.*}} : index
// CHECK-DAG: %[[MUL:.*]] = arith.muli %[[R2]], %{{.*}} : index
// CHECK-DAG: %[[ROW:.*]] = arith.remui %[[MUL]], %{{.*}} : index
// CHECK-DAG: %[[R3:.*]] = arith.remui %[[R1]], %{{.*}} : index
// CHECK-DAG: %[[ADD:.*]] = arith.addi %[[R3]], %{{.*}} : index
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%[[ROW]], %[[ADD]]] : !xegpu.mem_desc<32x32xf32>, index, index -> vector<2x1xf32>
// CHECK: %[[LANE_ID2:.*]] = gpu.lane_id
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%{{.*}}, %{{.*}}] : vector<2x1xf32>, !xegpu.mem_desc<32x32xf32>, index, index
gpu.func @load_store_matrix_2(%arg0: !xegpu.mem_desc<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = xegpu.load_matrix %arg0[%c0, %c1] <{layout = #xegpu.layout<lane_layout = [4, 4], lane_data = [2, 1]>}> : !xegpu.mem_desc<32x32xf32>, index, index -> vector<8x4xf32>
  xegpu.store_matrix %1, %arg0[%c0, %c1] <{layout = #xegpu.layout<lane_layout = [4, 4], lane_data = [2, 1]>}> : vector<8x4xf32>, !xegpu.mem_desc<32x32xf32>, index, index
  gpu.return
}
}

// -----
// load_matrix and store_matrix with subgroup_block_io (no coordinate computation)
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @load_store_matrix_3
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%{{.*}}, %{{.*}}] <{subgroup_block_io}>:
// CHECK-SAME: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<block = [16, 1], stride = [1, 32]>>, index, index -> vector<1x2xf32>
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%{{.*}}, %{{.*}}] <{subgroup_block_io}>:
// CHECK-SAME: vector<1x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<block = [16, 1], stride = [1, 32]>>, index, index
gpu.func @load_store_matrix_3(%arg0: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = xegpu.load_matrix %arg0[%c0, %c1] <{subgroup_block_io, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> :
    !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>, index, index -> vector<16x2xf32>
  xegpu.store_matrix %1, %arg0[%c0, %c1] <{subgroup_block_io, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> :
    vector<16x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>, index, index
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_step_slice
// CHECK:         %[[LANE_ID:.*]] = gpu.lane_id
// CHECK-DAG:     %[[C16:.*]] = arith.constant 16 : index
// CHECK:         %[[REM:.*]] = arith.remui %[[LANE_ID]], %[[C16]] : index
// CHECK:         %[[REM2:.*]] = arith.remui %[[REM]], %[[C16]]{{.*}} : index
// CHECK:         %[[VEC:.*]] = vector.from_elements %[[REM2]] : vector<1xindex>
gpu.func @vector_step_slice() {
  %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1]>, dims = [0, 1, 2]>} : vector<16xindex>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_step_slice_unit
// CHECK:         %[[VEC:.*]] = vector.from_elements %{{.*}} : vector<1xindex>
gpu.func @vector_step_slice_unit() {
  %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 1, 16], lane_data = [1, 1, 1, 1]>, dims = [0, 1, 3]>} : vector<1xindex>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_step_slice_multi_dist
// CHECK:         %[[LANE_ID:.*]] = gpu.lane_id
// CHECK:         %[[MULI:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:         %[[V0:.*]] = arith.remui %[[MULI]], %{{.*}} : index
// CHECK:         %[[SUM1:.*]] = arith.addi %[[MULI]], %{{.*}} : index
// CHECK:         %[[V2:.*]] = arith.remui %[[SUM1]], %{{.*}} : index
// CHECK:         %[[V1:.*]] = arith.addi %[[V0]], %{{.*}} : index
// CHECK:         %[[V3:.*]] = arith.addi %[[V2]], %{{.*}} : index
// CHECK:         %[[VEC:.*]] = vector.from_elements %[[V0]], %[[V1]], %[[V2]], %[[V3]] : vector<4xindex>
gpu.func @vector_step_slice_multi_dist() {
  %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [2, 4, 2], lane_data = [1, 2, 1]>, dims = [0, 2]>} : vector<16xindex>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_shapecast_rank_increasing
// CHECK:         %[[SC:.*]] = vector.shape_cast %{{.*}} : vector<1xf32> to vector<1x1xf32>
gpu.func @vector_shapecast_rank_increasing() {
  %cst = "some_op"()
    {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
    : () -> (vector<16xf32>)
  %cast = vector.shape_cast %cst
    {
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16xf32> to vector<1x16xf32>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_shapecast_rank_reducing
// CHECK:         %[[SC:.*]] = vector.shape_cast %{{.*}} : vector<1x1xf32> to vector<1xf32>
gpu.func @vector_shapecast_rank_reducing() {
  %cst = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> (vector<1x16xf32>)
  %cast = vector.shape_cast %cst
    {
      layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>
    }
    : vector<1x16xf32> to vector<16xf32>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_shapecast_rank_increasing_without_slicing_layout
// CHECK:         %[[SC:.*]] = vector.shape_cast %{{.*}} : vector<1xf32> to vector<1x1xf32>
gpu.func @vector_shapecast_rank_increasing_without_slicing_layout() {
  %cst = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> (vector<16xf32>)
  %cast = vector.shape_cast %cst
    {
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16xf32> to vector<1x16xf32>
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_broadcast_1d_to_2d
// CHECK: %[[SRC:.*]] = "some_op"()
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : vector<16xf16> to vector<1xf16>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] : vector<1xf16> to vector<16x1xf16>
gpu.func @vector_broadcast_1d_to_2d(%laneid: index) {
  %0 = "some_op"() {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : () -> vector<16xf16>
  %1 = vector.broadcast %0 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16xf16> to vector<16x16xf16>
  "some_use"(%1) : (vector<16x16xf16>) -> ()
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_broadcast_2d_to_3d
// CHECK: %[[SRC:.*]] = "some_op"()
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : vector<16x16xf16> to vector<16x1xf16>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] : vector<16x1xf16> to vector<1x16x1xf16>
gpu.func @vector_broadcast_2d_to_3d(%laneid: index) {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : () -> vector<16x16xf16>
  %1 = vector.broadcast %0 {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>} : vector<16x16xf16> to vector<1x16x16xf16>
  "some_use"(%1) : (vector<1x16x16xf16>) -> ()
  gpu.return
}
}

// -----
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_broadcast_2d_to_2d_noop
// CHECK: %[[SRC:.*]] = "some_op"()
// CHECK-NOT: vector.broadcast
gpu.func @vector_broadcast_2d_to_2d_noop(%laneid: index) {
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : () -> vector<16x1xf16>
  %1 = vector.broadcast %0 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x1xf16> to vector<16x16xf16>
  "some_use"(%1) : (vector<16x16xf16>) -> ()
  gpu.return
}
}

// -----
// Scalar to vector broadcast (with layout)
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_broadcast_scalar_to_vector
// CHECK: %[[SRC:.*]] = "some_op"()
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SRC]] : f16 to vector<16x1xf16>
gpu.func @vector_broadcast_scalar_to_vector(%laneid: index) {
  %0 = "some_op"() : () -> f16
  %1 = vector.broadcast %0 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : f16 to vector<16x16xf16>
  "some_use"(%1) : (vector<16x16xf16>) -> ()
  gpu.return
}
}

// -----
// Scalar to vector broadcast (no layout - uniform, should remain unchanged)
gpu.module @xevm_module {
// CHECK-LABEL: gpu.func @vector_broadcast_scalar_to_vector_uniform
// CHECK: %[[SRC:.*]] = "some_op"()
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SRC]] : f16 to vector<16x16xf16>
// CHECK: "some_use"(%[[BCAST]])
gpu.func @vector_broadcast_scalar_to_vector_uniform(%laneid: index) {
  %0 = "some_op"() : () -> f16
  %1 = vector.broadcast %0 : f16 to vector<16x16xf16>
  "some_use"(%1) : (vector<16x16xf16>) -> ()
  gpu.return
}
}
