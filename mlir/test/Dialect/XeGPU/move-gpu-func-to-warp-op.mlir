// RUN: mlir-opt -test-xegpu-move-func-to-warp-op -split-input-file --allow-unregistered-dialect %s | FileCheck %s

gpu.module @test {
gpu.func @empty()  {
  gpu.return
}
}

// CHECK-LABEL: gpu.func @empty() {
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:  }

// -----
gpu.module @test {
gpu.func @gemm(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0[%c0, %c0] : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1[%c0, %c0] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5[%c0, %c0] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}

// CHECK-LABEL: gpu.func @gemm(
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]: memref<8x16xf16>, %[[ARG1:[a-zA-Z0-9]+]]: memref<16x16xf16>,
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: memref<8x16xf32>) {
// CHECK:         %[[LANEID:.*]] = gpu.lane_id
// CHECK-NEXT:    gpu.warp_execute_on_lane_0(%[[LANEID]])[16]
// CHECK-SAME:      args(%[[ARG0]], %[[ARG1]], %[[ARG2]] : memref<8x16xf16>, memref<16x16xf16>, memref<8x16xf32>) {
// CHECK:           ^bb0(%[[ARG3:[a-zA-Z0-9]+]]: memref<8x16xf16>, %[[ARG4:[a-zA-Z0-9]+]]: memref<16x16xf16>,
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: memref<8x16xf32>):
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG3]] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT:      %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG4]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT:      %[[T3:.*]] = xegpu.load_nd %[[T1]][{{.*}}]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT:      %[[T4:.*]] = xegpu.load_nd %[[T2]][{{.*}}]  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT:      %[[T5:.*]] = xegpu.dpas %[[T3]], %[[T4]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT:      %[[T6:.*]] = xegpu.create_nd_tdesc %[[ARG5]] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT:      xegpu.store_nd %[[T5]], %[[T6]][%{{.*}}]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    gpu.return

// -----
gpu.module @test {
gpu.func @already_in_warp_op() {
  %laneid = gpu.lane_id
  gpu.warp_execute_on_lane_0(%laneid)[16] {
    "some_op"() : () -> ()
    gpu.yield
  }
  gpu.return
}
}

// CHECK-LABEL: gpu.func @already_in_warp_op() {
// CHECK:         %[[LANEID:.*]] = gpu.lane_id
// CHECK:         gpu.warp_execute_on_lane_0(%[[LANEID]])[16] {
// CHECK:           "some_op"() : () -> ()
// CHECK:         }
// CHECK:         gpu.return
