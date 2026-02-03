
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
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[LANE_RED:.*]] = vector.reduction <add>, %[[CAST:.*]] : vector<2xf32> into f32
// CHECK: %[[C16_1:.*]] = arith.constant 16 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[SHUFFLE1:.*]], %{{.*}} = gpu.shuffle  xor %[[LANE_RED]], %[[C1]], %[[C16_1]] : f32
// CHECK: %[[ADD1:.*]] = arith.addf %[[LANE_RED]], %[[SHUFFLE1]] : f32
// CHECK: %[[C16_2:.*]] = arith.constant 16 : i32
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[SHUFFLE2:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD1]], %[[C2]], %[[C16_2]] : f32
// CHECK: %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[SHUFFLE2]] : f32
// CHECK: %[[C16_3:.*]] = arith.constant 16 : i32
// CHECK: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[SHUFFLE3:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD2]], %[[C4]], %[[C16_3]] : f32
// CHECK: %[[ADD3:.*]] = arith.addf %[[ADD2]], %[[SHUFFLE3]] : f32
// CHECK: %[[C16_4:.*]] = arith.constant 16 : i32
// CHECK: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[SHUFFLE4:.*]], %{{.*}} = gpu.shuffle  xor %[[ADD3]], %[[C8]], %[[C16_4]] : f32
// CHECK: %[[ADD4:.*]] = arith.addf %[[ADD3]], %[[SHUFFLE4]] : f32
// CHECK: %[[FINAL:.*]] = arith.addf %[[ADD4]], %[[CST]] : f32
gpu.func @vector_reduction() {
  %acc = arith.constant 1.0 : f32
  %0 = "some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} : () -> vector<32xf32>
  %2 = vector.reduction <add>, %0, %acc : vector<32xf32> into f32
  gpu.return
}

}
