// RUN: mlir-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s --implicit-check-not='arith.{{(extui|extsi|trunci)}} {{.*}}xi4>' --implicit-check-not=arith.ori --implicit-check-not=xegpu.lane_shuffle


// Hoist only the i4-to-i8 part of the extension before the shuffle. Extending
// to i32 first would exceed the shuffle's 64-bit fragment limit. Starting with
// packed bytes also checks that narrow-type legalization unpacks the nibbles
// without repacking them after the shuffle.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_shape_cast_extui
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>
gpu.func @unpack_shape_cast_extui(%a: vector<2xi8>) -> vector<1x4xi32> {
  // CHECK: %[[MASK:.*]] = arith.constant dense<15> : vector<2xi8>
  // CHECK: %[[LOW:.*]] = arith.andi %[[SRC]], %[[MASK]] : vector<2xi8>
  // CHECK: %[[SHIFT:.*]] = arith.constant dense<4> : vector<2xi8>
  // CHECK: %[[HIGH:.*]] = arith.shrui %[[SRC]], %[[SHIFT]] : vector<2xi8>
  // CHECK: %[[BYTES:.*]] = vector.shuffle %[[LOW]], %[[HIGH]] [0, 2, 1, 3] : vector<2xi8>, vector<2xi8>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[BYTES]] : vector<4xi8> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[SHUFFLED]] : vector<4xi8> to vector<1x4xi8>
  // CHECK: %[[RES:.*]] = arith.extui %[[SHAPED]] : vector<1x4xi8> to vector<1x4xi32>
  // CHECK: gpu.return %[[RES]] : vector<1x4xi32>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles unpack : vector<4xi4>
  %shaped = vector.shape_cast %shuffled : vector<4xi4> to vector<1x4xi4>
  %res = arith.extui %shaped : vector<1x4xi4> to vector<1x4xi32>
  gpu.return %res : vector<1x4xi32>
}
}

// -----

// Signed widening must sign-extend each nibble before packing the byte vector,
// then preserve the signed extension from i8 to i32 after the shuffle.
gpu.module @test {
// CHECK-LABEL: gpu.func @pack_extsi
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>
gpu.func @pack_extsi(%a: vector<2xi8>) -> vector<4xi32> {
  // CHECK: %[[SHIFT:.*]] = arith.constant dense<4> : vector<2xi8>
  // CHECK: %[[LEFT:.*]] = arith.shli %[[SRC]], %[[SHIFT]] : vector<2xi8>
  // CHECK: %[[LOW:.*]] = arith.shrsi %[[LEFT]], %{{.*}} : vector<2xi8>
  // CHECK: %[[HIGH:.*]] = arith.shrsi %[[SRC]], %{{.*}} : vector<2xi8>
  // CHECK: %[[BYTES:.*]] = vector.shuffle %[[LOW]], %[[HIGH]] [0, 2, 1, 3] : vector<2xi8>, vector<2xi8>
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[BYTES]] : (vector<4xi8>) -> i32
  // CHECK: %[[SHUFFLED:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles pack : vector<4xi4>
  %res = arith.extsi %shuffled : vector<4xi4> to vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// An extension whose destination is already i8 disappears entirely.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_extui_i8
gpu.func @unpack_extui_i8(%a: vector<2xi8>) -> vector<4xi8> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %{{.*}} : vector<4xi8> to i32
  // CHECK: %[[RES:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  // CHECK-NEXT: gpu.return %[[RES]] : vector<4xi8>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles unpack : vector<4xi4>
  %res = arith.extui %shuffled : vector<4xi4> to vector<4xi8>
  gpu.return %res : vector<4xi8>
}
}

// -----

// The smallest supported fragment widens to 16 bits. Signed extension to i8
// needs no remaining cast either.
gpu.module @test {
// CHECK-LABEL: gpu.func @pack_two_extsi_i8
gpu.func @pack_two_extsi_i8(%a: vector<1xi8>) -> vector<2xi8> {
  // CHECK: arith.shrsi {{.*}} : vector<1xi8>
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %{{.*}} : (vector<2xi8>) -> i16
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[PACKED]] : i16 to vector<2xi8>
  // CHECK-NEXT: gpu.return %[[RES]] : vector<2xi8>
  %nibbles = vector.bitcast %a : vector<1xi8> to vector<2xi4>
  %shuffled = xegpu.lane_shuffle %nibbles pack : vector<2xi4>
  %res = arith.extsi %shuffled : vector<2xi4> to vector<2xi8>
  gpu.return %res : vector<2xi8>
}
}

// -----

// Eight elements reach the byte shuffle's 64-bit limit. Hoist through a slice
// followed by a shape cast, keeping the slice and final extension after it.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_eight_slice_shape_extui
gpu.func @unpack_eight_slice_shape_extui(%a: vector<4xi8>) -> vector<1x4xi32> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %{{.*}} : vector<8xi8> to i64
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i64) -> vector<8xi8>
  // CHECK: %[[SLICE:.*]] = vector.extract_strided_slice %[[SHUFFLED]] offsets = [2], sizes = [4], strides = [1] : vector<8xi8> to vector<4xi8>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[SLICE]] : vector<4xi8> to vector<1x4xi8>
  // CHECK: %[[RES:.*]] = arith.extui %[[SHAPED]] : vector<1x4xi8> to vector<1x4xi32>
  // CHECK: gpu.return %[[RES]] : vector<1x4xi32>
  %nibbles = vector.bitcast %a : vector<4xi8> to vector<8xi4>
  %shuffled = xegpu.lane_shuffle %nibbles unpack : vector<8xi4>
  %slice = vector.extract_strided_slice %shuffled offsets = [2], sizes = [4], strides = [1] : vector<8xi4> to vector<4xi4>
  %shaped = vector.shape_cast %slice : vector<4xi4> to vector<1x4xi4>
  %res = arith.extui %shaped : vector<1x4xi4> to vector<1x4xi32>
  gpu.return %res : vector<1x4xi32>
}
}

// -----

// Exercise the remaining supported movers in a single-use chain: transpose,
// vector-valued extract, and broadcast. None may force nibble repacking.
gpu.module @test {
// CHECK-LABEL: gpu.func @pack_transpose_extract_broadcast_extsi
gpu.func @pack_transpose_extract_broadcast_extsi(%a: vector<2xi8>) -> vector<3x2xi16> {
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %{{.*}} : (vector<4xi8>) -> i32
  // CHECK: %[[SHUFFLED:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<4xi8>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[SHUFFLED]] : vector<4xi8> to vector<2x2xi8>
  // CHECK: %[[TRANSPOSED:.*]] = vector.transpose %[[SHAPED]], [1, 0] : vector<2x2xi8> to vector<2x2xi8>
  // CHECK: %[[ROW:.*]] = vector.extract %[[TRANSPOSED]][1] : vector<2xi8> from vector<2x2xi8>
  // CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[ROW]] : vector<2xi8> to vector<3x2xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[BROADCAST]] : vector<3x2xi8> to vector<3x2xi16>
  // CHECK: gpu.return %[[RES]] : vector<3x2xi16>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles pack : vector<4xi4>
  %shaped = vector.shape_cast %shuffled : vector<4xi4> to vector<2x2xi4>
  %transposed = vector.transpose %shaped, [1, 0] : vector<2x2xi4> to vector<2x2xi4>
  %row = vector.extract %transposed[1] : vector<2xi4> from vector<2x2xi4>
  %broadcast = vector.broadcast %row : vector<2xi4> to vector<3x2xi4>
  %res = arith.extsi %broadcast : vector<3x2xi4> to vector<3x2xi16>
  gpu.return %res : vector<3x2xi16>
}
}

// -----

// Distribution can split a broadcast into several slices of one shared
// shuffle. Linearization turns the one-element slices into scalar extracts.
// Widen the shared fragment once, before either representation of the fan-out.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_shared_broadcast_extui
// CHECK: %[[WIDE:.*]] = xevm.bitcast_shuffle {{.*}} : (i32) -> vector<4xi8>
// CHECK: %[[SHAPE:.*]] = vector.shape_cast %[[WIDE]] : vector<4xi8> to vector<1x4xi8>
// CHECK: %[[SLICE0:.*]] = vector.extract_strided_slice %[[SHAPE]] offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : vector<1x4xi8> to vector<1x1xi8>
// CHECK: %[[SLICE1:.*]] = vector.extract_strided_slice %[[SHAPE]] offsets = [0, 1], sizes = [1, 1], strides = [1, 1] : vector<1x4xi8> to vector<1x1xi8>
// CHECK: vector.broadcast %[[SLICE0]] : vector<1x1xi8> to vector<1x16xi8>
// CHECK: vector.broadcast %[[SLICE1]] : vector<1x1xi8> to vector<1x16xi8>
// CHECK: arith.extui {{.*}} : vector<1x16xi8> to vector<1x16xi32>
// CHECK: arith.extui {{.*}} : vector<1x16xi8> to vector<1x16xi32>

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

// -----

// Compatible extensions with different destination widths share one widened
// shuffle. One extension can disappear without invalidating the other.
gpu.module @test {
// CHECK-LABEL: gpu.func @pack_shared_extsi
// CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle {{.*}} : (vector<4xi8>) -> i32
// CHECK: %[[WIDE:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<4xi8>
// CHECK: %[[EXT:.*]] = arith.extsi %[[WIDE]] : vector<4xi8> to vector<4xi32>
// CHECK: gpu.return %[[WIDE]], %[[EXT]] : vector<4xi8>, vector<4xi32>
gpu.func @pack_shared_extsi(%a: vector<2xi8>) -> (vector<4xi8>, vector<4xi32>) {
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shuffled = xegpu.lane_shuffle %nibbles pack : vector<4xi4>
  %r0 = arith.extsi %shuffled : vector<4xi4> to vector<4xi8>
  %r1 = arith.extsi %shuffled : vector<4xi4> to vector<4xi32>
  gpu.return %r0, %r1 : vector<4xi8>, vector<4xi32>
}
}
