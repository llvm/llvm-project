
// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=pvc' --allow-unregistered-dialect \
// RUN: --test-xegpu-sg-to-wi-distribute-experimental --split-input-file %s | FileCheck %s

// RUN: mlir-opt --allow-unregistered-dialect \
// RUN: --test-xegpu-sg-to-wi-distribute-experimental="enable-rewrite-multi-reduction-to-reductions"  \
// RUN: --split-input-file  %s | FileCheck --check-prefix=CHECK-REWRITE %s



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


// CHECK-REWRITE-LABEL: gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction
// CHECK-REWRITE-DAG:     %[[SRC:.*]] = "some_def"() {layout_result_0 =
// CHECK-REWRITE-SAME:      #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : () -> vector<2x16xf32>
// CHECK-REWRITE-DAG:     %[[ACC:.*]] = arith.constant
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      dense<0.000000e+00> : vector<2xf32>
// CHECK-REWRITE-DAG:     %[[ZERO:.*]] = arith.constant
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      dense<0.000000e+00> : vector<2xf32>
// CHECK-REWRITE:         %[[SLICE0:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
// CHECK-REWRITE-SAME:       offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<2x16xf32> to vector<1x16xf32>
// CHECK-REWRITE-NEXT:    %[[CAST0:.*]] = vector.shape_cast %[[SLICE0]]
// CHECK-REWRITE-SAME:      {{{.*}}, layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      : vector<1x16xf32> to vector<16xf32>
// CHECK-REWRITE-NEXT:    %[[ACC0:.*]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[RED0:.*]] = vector.reduction <add>, %[[CAST0]], %[[ACC0]] : vector<16xf32> into f32
// CHECK-REWRITE-NEXT:    %[[INS0:.*]] = vector.insert %[[RED0]], %[[ZERO]] [0]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      : f32 into vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[SLICE1:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
// CHECK-REWRITE-SAME:       offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<2x16xf32> to vector<1x16xf32>
// CHECK-REWRITE-NEXT:    %[[CAST1:.*]] = vector.shape_cast %[[SLICE1]]
// CHECK-REWRITE-SAME:      {{{.*}}, layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      : vector<1x16xf32> to vector<16xf32>
// CHECK-REWRITE-NEXT:    %[[ACC1:.*]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[RED1:.*]] = vector.reduction <add>, %[[CAST1]], %[[ACC1]] : vector<16xf32> into f32
// CHECK-REWRITE-NEXT:    %[[INS1:.*]] = vector.insert %[[RED1]], %[[INS0]] [1]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
// CHECK-REWRITE-SAME:      : f32 into vector<2xf32>
gpu.func @vector_multi_reduction_dim1_distributed_dim1_reduction(%laneid: index) {
  %c0 = arith.constant 0 : index
    %src = "some_def"()
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : () -> (vector<2x16xf32>)
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

// CHECK-REWRITE-LABEL: gpu.func @vector_multi_reduction_dim0_distributed_dim0_reduction
// CHECK-REWRITE-DAG:     %[[SRC:.*]] = "some_def"() {layout_result_0 =
// CHECK-REWRITE-SAME:      #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>} : () -> vector<16x2xf32>
// CHECK-REWRITE-DAG:     %[[ACC:.*]] = arith.constant
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
// CHECK-REWRITE-SAME:      dense<0.000000e+00> : vector<2xf32>
// CHECK-REWRITE-DAG:     %[[ZERO:.*]] = arith.constant
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
// CHECK-REWRITE-SAME:      dense<0.000000e+00> : vector<2xf32>
// CHECK-REWRITE:         %[[SLICE0:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
// CHECK-REWRITE-SAME:       offsets = [0, 0], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK-REWRITE-NEXT:    %[[CAST0:.*]] = vector.shape_cast %[[SLICE0]]
// CHECK-REWRITE-SAME:      {{.*}}, layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
// CHECK-REWRITE-SAME:      : vector<16x1xf32> to vector<16xf32>
// CHECK-REWRITE-NEXT:    %[[ACC0:.*]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[RED0:.*]] = vector.reduction <add>, %[[CAST0]], %[[ACC0]] : vector<16xf32> into f32
// CHECK-REWRITE-NEXT:    %[[INS0:.*]] = vector.insert %[[RED0]], %[[ZERO]] [0]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
// CHECK-REWRITE-SAME:      : f32 into vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[SLICE1:.*]] = vector.extract_strided_slice %[[SRC]]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
// CHECK-REWRITE-SAME:       offsets = [0, 1], sizes = [16, 1], strides = [1, 1]} : vector<16x2xf32> to vector<16x1xf32>
// CHECK-REWRITE-NEXT:    %[[CAST1:.*]] = vector.shape_cast %[[SLICE1]]
// CHECK-REWRITE-SAME:      {{{.*}}, layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>
// CHECK-REWRITE-SAME:      : vector<16x1xf32> to vector<16xf32>
// CHECK-REWRITE-NEXT:    %[[ACC1:.*]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
// CHECK-REWRITE-NEXT:    %[[RED1:.*]] = vector.reduction <add>, %[[CAST1]], %[[ACC1]] : vector<16xf32> into f32
// CHECK-REWRITE-NEXT:    %[[INS1:.*]] = vector.insert %[[RED1]], %[[INS0]] [1]
// CHECK-REWRITE-SAME:      {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>, dims = [0]>}
// CHECK-REWRITE-SAME:      : f32 into vector<2xf32>
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
