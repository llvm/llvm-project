// RUN: mlir-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s --implicit-check-not=xegpu.lane_shuffle

// The shuffle moves the narrow value, and the extension comes after it. As
// vector<4xi32>, the four values are 128 bits per lane. The lowering takes at
// most 64. As vector<4xi8>, they are 32 bits. The i4 value is extended to i8
// first, because the lowering takes no element narrower than 8 bits. The
// shape_cast between the extension and the shuffle is rebuilt on i8.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extui_i4
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>
gpu.func @unpack_after_extui_i4(%a: vector<2xi8>) -> vector<4xi32> {
  // CHECK: %[[NIBBLES:.*]] = vector.bitcast %[[SRC]] : vector<2xi8> to vector<4xi4>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[NIBBLES]] : vector<4xi4> to vector<1x4xi4>
  // CHECK: %[[BYTES:.*]] = vector.bitcast %[[SHAPED]] : vector<1x4xi4> to vector<1x2xi8>
  // CHECK: %[[MASK:.*]] = arith.constant dense<15> : vector<1x2xi8>
  // CHECK: %[[LOW:.*]] = arith.andi %[[BYTES]], %[[MASK]] : vector<1x2xi8>
  // CHECK: %[[SHIFT:.*]] = arith.constant dense<4> : vector<1x2xi8>
  // CHECK: %[[HIGH:.*]] = arith.shrui %[[BYTES]], %[[SHIFT]] : vector<1x2xi8>
  // CHECK: %[[WIDE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<1x2xi8> -> vector<1x4xi8>
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[WIDE]] : vector<1x4xi8> to vector<4xi8>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[FLAT]] : vector<4xi8> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extui %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %shaped = vector.shape_cast %nibbles : vector<4xi4> to vector<1x4xi4>
  %wide = arith.extui %shaped : vector<1x4xi4> to vector<1x4xi32>
  %flat = vector.shape_cast %wide : vector<1x4xi32> to vector<4xi32>
  %res = xegpu.lane_shuffle %flat unpack : vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// A signed i4 value is sign-extended twice: to i8 before the shuffle and to
// i32 after it.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extsi_i4
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>
gpu.func @unpack_after_extsi_i4(%a: vector<2xi8>) -> vector<4xi32> {
  // CHECK: %[[C4:.*]] = arith.constant dense<4> : vector<2xi8>
  // CHECK: %[[UP:.*]] = arith.shli %[[SRC]], %[[C4]] : vector<2xi8>
  // CHECK: %[[LOW_SHIFT:.*]] = arith.constant dense<4> : vector<2xi8>
  // CHECK: %[[LOW:.*]] = arith.shrsi %[[UP]], %[[LOW_SHIFT]] : vector<2xi8>
  // CHECK: %[[HIGH_SHIFT:.*]] = arith.constant dense<4> : vector<2xi8>
  // CHECK: %[[HIGH:.*]] = arith.shrsi %[[SRC]], %[[HIGH_SHIFT]] : vector<2xi8>
  // CHECK: %[[WIDE:.*]] = vector.shuffle %[[LOW]], %[[HIGH]] [0, 2, 1, 3] : vector<2xi8>, vector<2xi8>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[WIDE]] : vector<4xi8> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %nibbles = vector.bitcast %a : vector<2xi8> to vector<4xi4>
  %wide = arith.extsi %nibbles : vector<4xi4> to vector<4xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// An i8 value is shuffled as it is.
gpu.module @test {
// CHECK-LABEL: gpu.func @pack_after_extsi_i8
// CHECK-SAME: %[[SRC:.*]]: vector<4xi8>
gpu.func @pack_after_extsi_i8(%a: vector<4xi8>) -> vector<4xi32> {
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[SRC]] : (vector<4xi8>) -> i32
  // CHECK: %[[SHUFFLED:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %wide = arith.extsi %a : vector<4xi8> to vector<4xi32>
  %res = xegpu.lane_shuffle %wide pack : vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// The vector<2xi32> shuffle already fits the lowering. It still moves the
// narrow vector<2xi16> value.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extui_i16
// CHECK-SAME: %[[SRC:.*]]: vector<2xi16>
gpu.func @unpack_after_extui_i16(%a: vector<2xi16>) -> vector<2xi32> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[SRC]] : vector<2xi16> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<2xi16>
  // CHECK: %[[RES:.*]] = arith.extui %[[SHUFFLED]] : vector<2xi16> to vector<2xi32>
  // CHECK: gpu.return %[[RES]] : vector<2xi32>
  %wide = arith.extui %a : vector<2xi16> to vector<2xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<2xi32>
  gpu.return %res : vector<2xi32>
}
}

// -----

// The lowering takes only 8, 16, 32 or 64-bit elements, so an i12 value is
// extended to i16 before the shuffle.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extsi_i12
// CHECK-SAME: %[[SRC:.*]]: vector<2xi12>
gpu.func @unpack_after_extsi_i12(%a: vector<2xi12>) -> vector<2xi32> {
  // CHECK: %[[HALF:.*]] = arith.extsi %[[SRC]] : vector<2xi12> to vector<2xi16>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[HALF]] : vector<2xi16> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<2xi16>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<2xi16> to vector<2xi32>
  // CHECK: gpu.return %[[RES]] : vector<2xi32>
  %wide = arith.extsi %a : vector<2xi12> to vector<2xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<2xi32>
  gpu.return %res : vector<2xi32>
}
}

// -----

// An i1 value is extended to i8 before the shuffle.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extsi_i1
// CHECK-SAME: %[[SRC:.*]]: vector<4xi1>
gpu.func @unpack_after_extsi_i1(%a: vector<4xi1>) -> vector<4xi32> {
  // CHECK: %[[BYTES:.*]] = arith.extsi %[[SRC]] : vector<4xi1> to vector<4xi8>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[BYTES]] : vector<4xi8> to i32
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %wide = arith.extsi %a : vector<4xi1> to vector<4xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// An i24 value would be extended to i32, which is no narrower than the result.
// So nothing changes, and the shuffle moves the i32 value.
gpu.module @test {
// CHECK-LABEL: gpu.func @unpack_after_extui_i24
// CHECK-SAME: %[[SRC:.*]]: vector<2xi24>
gpu.func @unpack_after_extui_i24(%a: vector<2xi24>) -> vector<2xi32> {
  // CHECK: %[[WIDE:.*]] = arith.extui %[[SRC]] : vector<2xi24> to vector<2xi32>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[WIDE]] : vector<2xi32> to i64
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i64) -> vector<2xi32>
  // CHECK: gpu.return %[[SHUFFLED]] : vector<2xi32>
  %wide = arith.extui %a : vector<2xi24> to vector<2xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<2xi32>
  gpu.return %res : vector<2xi32>
}
}

// -----

// A transpose, a slice and a shape_cast between the extension and the shuffle
// are rebuilt on i8, with the same permutation, offsets and shapes.
gpu.module @test {
// CHECK-LABEL: gpu.func @movers_after_extsi
// CHECK-SAME: %[[SRC:.*]]: vector<2x4xi8>
gpu.func @movers_after_extsi(%a: vector<2x4xi8>) -> vector<4xi32> {
  // CHECK: %[[T:.*]] = vector.transpose %[[SRC]], [1, 0] : vector<2x4xi8> to vector<4x2xi8>
  // CHECK: %[[S:.*]] = vector.extract_strided_slice %[[T]] offsets = [2, 0], sizes = [2, 2], strides = [1, 1] : vector<4x2xi8> to vector<2x2xi8>
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[S]] : vector<2x2xi8> to vector<4xi8>
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[FLAT]] : (vector<4xi8>) -> i32
  // CHECK: %[[SHUFFLED:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<4xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<4xi8> to vector<4xi32>
  // CHECK: gpu.return %[[RES]] : vector<4xi32>
  %wide = arith.extsi %a : vector<2x4xi8> to vector<2x4xi32>
  %t = vector.transpose %wide, [1, 0] : vector<2x4xi32> to vector<4x2xi32>
  %s = vector.extract_strided_slice %t offsets = [2, 0], sizes = [2, 2], strides = [1, 1] : vector<4x2xi32> to vector<2x2xi32>
  %flat = vector.shape_cast %s : vector<2x2xi32> to vector<4xi32>
  %res = xegpu.lane_shuffle %flat pack : vector<4xi32>
  gpu.return %res : vector<4xi32>
}
}

// -----

// A vector.extract with dynamic positions gives an i8 scalar, at the same
// positions. The broadcast of it is rebuilt on i8.
gpu.module @test {
// CHECK-LABEL: gpu.func @extract_and_broadcast
// CHECK-SAME: %[[SRC:.*]]: vector<2x2xi8>, %[[ROW:.*]]: index, %[[COL:.*]]: index
gpu.func @extract_and_broadcast(%a: vector<2x2xi8>, %row: index, %col: index) -> vector<2xi32> {
  // CHECK: %[[ONE:.*]] = vector.extract %[[SRC]][%[[ROW]], %[[COL]]] : i8 from vector<2x2xi8>
  // CHECK: %[[PAIR:.*]] = vector.broadcast %[[ONE]] : i8 to vector<2xi8>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[PAIR]] : vector<2xi8> to i16
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i16) -> vector<2xi8>
  // CHECK: %[[RES:.*]] = arith.extsi %[[SHUFFLED]] : vector<2xi8> to vector<2xi32>
  // CHECK: gpu.return %[[RES]] : vector<2xi32>
  %wide = arith.extsi %a : vector<2x2xi8> to vector<2x2xi32>
  %one = vector.extract %wide[%row, %col] : i32 from vector<2x2xi32>
  %pair = vector.broadcast %one : i32 to vector<2xi32>
  %res = xegpu.lane_shuffle %pair unpack : vector<2xi32>
  gpu.return %res : vector<2xi32>
}
}

// -----

// The extension has another use, so nothing changes.
gpu.module @test {
// CHECK-LABEL: gpu.func @extension_with_other_use
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>
gpu.func @extension_with_other_use(%a: vector<2xi8>) -> (vector<2xi32>, vector<2xi32>) {
  // CHECK: %[[WIDE:.*]] = arith.extui %[[SRC]] : vector<2xi8> to vector<2xi32>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[WIDE]] : vector<2xi32> to i64
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i64) -> vector<2xi32>
  // CHECK: gpu.return %[[SHUFFLED]], %[[WIDE]] : vector<2xi32>, vector<2xi32>
  %wide = arith.extui %a : vector<2xi8> to vector<2xi32>
  %res = xegpu.lane_shuffle %wide unpack : vector<2xi32>
  gpu.return %res, %wide : vector<2xi32>, vector<2xi32>
}
}

// -----

// The shape_cast between the extension and the shuffle has another use, so
// nothing changes.
gpu.module @test {
// CHECK-LABEL: gpu.func @mover_with_other_use
// CHECK-SAME: %[[SRC:.*]]: vector<1x2xi8>
gpu.func @mover_with_other_use(%a: vector<1x2xi8>) -> (vector<2xi32>, vector<2xi32>) {
  // CHECK: %[[WIDE:.*]] = arith.extui %[[SRC]] : vector<1x2xi8> to vector<1x2xi32>
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[WIDE]] : vector<1x2xi32> to vector<2xi32>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[FLAT]] : vector<2xi32> to i64
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i64) -> vector<2xi32>
  // CHECK: gpu.return %[[SHUFFLED]], %[[FLAT]] : vector<2xi32>, vector<2xi32>
  %wide = arith.extui %a : vector<1x2xi8> to vector<1x2xi32>
  %flat = vector.shape_cast %wide : vector<1x2xi32> to vector<2xi32>
  %res = xegpu.lane_shuffle %flat unpack : vector<2xi32>
  gpu.return %res, %flat : vector<2xi32>, vector<2xi32>
}
}

// -----

// An op that is not a data-movement op sits between the extension and the
// shuffle, so nothing changes.
gpu.module @test {
// CHECK-LABEL: gpu.func @other_op_between
// CHECK-SAME: %[[SRC:.*]]: vector<2xi8>, %[[B:.*]]: vector<2xi32>
gpu.func @other_op_between(%a: vector<2xi8>, %b: vector<2xi32>) -> vector<2xi32> {
  // CHECK: %[[WIDE:.*]] = arith.extui %[[SRC]] : vector<2xi8> to vector<2xi32>
  // CHECK: %[[SUM:.*]] = arith.addi %[[WIDE]], %[[B]] : vector<2xi32>
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[SUM]] : vector<2xi32> to i64
  // CHECK: %[[SHUFFLED:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i64) -> vector<2xi32>
  // CHECK: gpu.return %[[SHUFFLED]] : vector<2xi32>
  %wide = arith.extui %a : vector<2xi8> to vector<2xi32>
  %sum = arith.addi %wide, %b : vector<2xi32>
  %res = xegpu.lane_shuffle %sum unpack : vector<2xi32>
  gpu.return %res : vector<2xi32>
}
}
