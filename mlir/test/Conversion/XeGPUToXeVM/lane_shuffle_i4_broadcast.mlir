// RUN: mlir-opt -xegpu-vector-linearize -convert-xegpu-to-xevm %s | FileCheck %s --implicit-check-not=xegpu.lane_shuffle --implicit-check-not='arith.extui {{.*}}xi4>'

// Lane distribution splits a broadcast into slices of a shared shuffle.
// Linearization turns these one-element slices into scalar extracts before
// XeGPU conversion. Widen the fragment once, then extract and broadcast bytes.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_shared_broadcast_extui
// CHECK: %[[WIDE:.*]] = xevm.bitcast_shuffle {{.*}} : (i32) -> vector<4xi8>
// CHECK: %[[E0:.*]] = vector.extract %[[WIDE]][0] : i8 from vector<4xi8>
// CHECK: %[[B0:.*]] = vector.broadcast %[[E0]] : i8 to vector<16xi8>
// CHECK: %[[E1:.*]] = vector.extract %[[WIDE]][1] : i8 from vector<4xi8>
// CHECK: %[[B1:.*]] = vector.broadcast %[[E1]] : i8 to vector<16xi8>
// CHECK: arith.extui %[[B0]] : vector<16xi8> to vector<16xi32>
// CHECK: arith.extui %[[B1]] : vector<16xi8> to vector<16xi32>
gpu.func @unpack_shared_broadcast_extui(%a: vector<2xi8>) -> (vector<1x16xi32>, vector<1x16xi32>) {
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles unpack : vector<4xi4>
  %shaped = vector.shape_cast %shuffled : vector<4xi4> to vector<1x4xi4>
  %slice0 = vector.extract_strided_slice %shaped offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : vector<1x4xi4> to vector<1x1xi4>
  %slice1 = vector.extract_strided_slice %shaped offsets = [0, 1], sizes = [1, 1], strides = [1, 1] : vector<1x4xi4> to vector<1x1xi4>
  %b0 = vector.broadcast %slice0 : vector<1x1xi4> to vector<1x16xi4>
  %b1 = vector.broadcast %slice1 : vector<1x1xi4> to vector<1x16xi4>
  %r0 = arith.extui %b0 : vector<1x16xi4> to vector<1x16xi32>
  %r1 = arith.extui %b1 : vector<1x16xi4> to vector<1x16xi32>
  gpu.return %r0, %r1 : vector<1x16xi32>, vector<1x16xi32>
}
}
