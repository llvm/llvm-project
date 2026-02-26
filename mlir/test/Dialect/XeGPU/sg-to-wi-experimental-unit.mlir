
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

// CHECK-LABEL: gpu.func @vector_extract_from_2d_idx2
// CHECK: %[[EXT:.*]] = vector.extract %{{.*}}[2] : vector<1xf32> from vector<8x1xf32>
gpu.func @vector_extract_from_2d_idx2() {
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

// CHECK-LABEL: gpu.func @vector_insert_into_2d_idx2
// CHECK: %[[INS:.*]] = vector.insert %{{.*}}, %{{.*}}[2] : vector<1xf32> into vector<8x1xf32>
gpu.func @vector_insert_into_2d_idx2() {
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

// extract_strided_slice: distributed dim fully extracted
// Source: vector<24x16xf32> layout [1,16] → distributed to vector<24x1xf32>
// Result: vector<8x16xf32> layout [1,16] → distributed to vector<8x1xf32>
// Offsets [8,0] sizes [8,16] → [8,0] sizes [8,1]
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_distributed_dim_fully_extracted
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<24x1xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_distributed_dim_fully_extracted() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8, 0], sizes = [8, 16], strides = [1, 1],
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x16xf32> to vector<8x16xf32>
  gpu.return
}

// extract_strided_slice: non-distributed (source already has unit dim)
// Source: vector<24x1xf32> layout [1,16] → stays vector<24x1xf32>
// Result: vector<8x1xf32> layout [1,16] → stays vector<8x1xf32>
// Offsets and sizes unchanged
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_non_distributed
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<24x1xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_non_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x1xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8, 0], sizes = [8, 1], strides = [1, 1],
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x1xf32> to vector<8x1xf32>
  gpu.return
}

// extract_strided_slice: inner distributed (dim 1 distributed)
// Source: vector<24x64xf32> layout [1,16] → distributed to vector<24x4xf32>
// Result: vector<8x16xf32> layout [1,16] → distributed to vector<8x1xf32>
// Offsets [8,48] → [8, 48/16=3], sizes [8,16] → [8,1]
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_inner_distributed
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 3], sizes = [8, 1], strides = [1, 1]} : vector<24x4xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_inner_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x64xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8, 48], sizes = [8, 16], strides = [1, 1],
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x64xf32> to vector<8x16xf32>
  gpu.return
}

// extract_strided_slice: outer distributed (dim 0 distributed)
// Source: vector<32x16xf32> layout [16,1] → distributed to vector<2x16xf32>
// Result: vector<16x16xf32> layout [16,1] → distributed to vector<1x16xf32>
// Offsets [16] padded to [16,0] → [16/16=1, 0], sizes [16] padded to [16,16] → [1,16]
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_outer_distributed
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<2x16xf32> to vector<1x16xf32>
gpu.func @vector_extract_strided_slice_outer_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
    : () -> vector<32x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [16], sizes = [16], strides = [1],
      layout_operand_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
    }
    : vector<32x16xf32> to vector<16x16xf32>
  gpu.return
}

// extract_strided_slice: 1D distributed
// Source: vector<64xf32> layout [16] → distributed to vector<4xf32>
// Result: vector<32xf32> layout [16] → distributed to vector<2xf32>
// Offsets [16] → [16/16=1], sizes [32] → [2]
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_1d
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
gpu.func @vector_extract_strided_slice_1d() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
    : () -> vector<64xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [16], sizes = [32], strides = [1],
      layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
    }
    : vector<64xf32> to vector<32xf32>
  gpu.return
}

// extract_strided_slice: partial offsets (offsets rank < source rank)
// Source: vector<24x16xf32> layout [1,16] → distributed to vector<24x1xf32>
// Result: vector<8x16xf32> layout [1,16] → distributed to vector<8x1xf32>
// Offsets [8] padded to [8,0], sizes [8] padded to [8,16] → [8,1]
// CHECK-LABEL: gpu.func @vector_extract_strided_slice_partial_offsets
// CHECK: %[[ESS:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<24x1xf32> to vector<8x1xf32>
gpu.func @vector_extract_strided_slice_partial_offsets() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<24x16xf32>
  %1 = vector.extract_strided_slice %0 { offsets = [8], sizes = [8], strides = [1],
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<24x16xf32> to vector<8x16xf32>
  gpu.return
}

// insert_strided_slice: distributed dim fully inserted (dim 1 distributed)
// Source: vector<16x16xf32> layout [1,16] → distributed to vector<16x1xf32>
// Dest: vector<64x16xf32> layout [1,16] → distributed to vector<64x1xf32>
// Offsets [24,0] → [24,0] (offset 0 / 16 = 0)
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
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<64x16xf32>
  gpu.return
}

// insert_strided_slice: non-distributed (types already have unit dim)
// Source: vector<16x1xf32> layout [1,16] → stays vector<16x1xf32>
// Dest: vector<64x1xf32> layout [1,16] → stays vector<64x1xf32>
// Offsets unchanged
// CHECK-LABEL: gpu.func @vector_insert_strided_slice_non_distributed
// CHECK: %[[ISS:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [24, 0], strides = [1, 1]} : vector<16x1xf32> into vector<64x1xf32>
gpu.func @vector_insert_strided_slice_non_distributed() {
  %0 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<16x1xf32>
  %1 = "some_op"()
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : () -> vector<64x1xf32>
  %2 = vector.insert_strided_slice %0, %1 { offsets = [24, 0], strides = [1, 1],
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16x1xf32> into vector<64x1xf32>
  gpu.return
}

// insert_strided_slice: inner distributed (dim 1 distributed)
// Source: vector<16x16xf32> layout [1,16] → distributed to vector<16x1xf32>
// Dest: vector<64x32xf32> layout [1,16] → distributed to vector<64x2xf32>
// Offsets [24,16] → [24, 16/16=1]
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
      layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<64x32xf32>
  gpu.return
}

// insert_strided_slice: outer distributed (dim 0 distributed)
// Source: vector<16x16xf32> layout [16,1] → distributed to vector<1x16xf32>
// Dest: vector<48x32xf32> layout [16,1] → distributed to vector<3x32xf32>
// Offsets [32,4] → [32/16=2, 4]
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
      layout_operand_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
    }
    : vector<16x16xf32> into vector<48x32xf32>
  gpu.return
}

// insert_strided_slice: 1D distributed
// Source: vector<16xf32> layout [16] → distributed to vector<1xf32>
// Dest: vector<48xf32> layout [16] → distributed to vector<3xf32>
// Offsets [16] → [16/16=1]
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
      layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>
    }
    : vector<16xf32> into vector<48xf32>
  gpu.return
}

// insert_strided_slice: different ranks (1D source into 2D dest)
// Source: vector<16xf32> layout [16] → distributed to vector<1xf32>
// Dest: vector<64x16xf32> layout [1,16] → distributed to vector<64x1xf32>
// Distributed dim 1, sourceDistDim = 1 - (2-1) = 0
// Offsets [13,0] → [13, 0/16=0]
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
      layout_operand_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>,
      layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    : vector<16xf32> into vector<64x16xf32>
  gpu.return
}
}
