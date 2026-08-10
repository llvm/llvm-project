// RUN: mlir-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_pack_i16
// CHECK-SAME:  %[[SRC:.*]]: vector<2xi16>
gpu.func @lane_shuffle_pack_i16(%a: vector<2xi16>) -> vector<2xi16> {
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[SRC]] : (vector<2xi16>) -> i32
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[PACKED]] : i32 to vector<2xi16>
  // CHECK: gpu.return %[[RES]]
  %0 = xegpu.lane_shuffle %a pack : vector<2xi16>
  gpu.return %0 : vector<2xi16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_unpack_i16
// CHECK-SAME:  %[[SRC:.*]]: vector<2xi16>
gpu.func @lane_shuffle_unpack_i16(%a: vector<2xi16>) -> vector<2xi16> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[SRC]] : vector<2xi16> to i32
  // CHECK: %[[RES:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<2xi16>
  // CHECK: gpu.return %[[RES]]
  %0 = xegpu.lane_shuffle %a unpack : vector<2xi16>
  gpu.return %0 : vector<2xi16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_pack_f16
// CHECK-SAME:  %[[SRC:.*]]: vector<4xf16>
gpu.func @lane_shuffle_pack_f16(%a: vector<4xf16>) -> vector<4xf16> {
  // The shuffle only takes integers, so the fragment is bitcast to a same-width
  // integer vector on the way in and back to f16 on the way out.
  // CHECK: %[[BITS:.*]] = llvm.bitcast %[[SRC]] : vector<4xf16> to vector<4xi16>
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[BITS]] : (vector<4xi16>) -> i64
  // CHECK: llvm.bitcast %[[PACKED]] : i64 to vector<4xf16>
  %0 = xegpu.lane_shuffle %a pack : vector<4xf16>
  gpu.return %0 : vector<4xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_pack_bf16
// CHECK-SAME:  %[[SRC:.*]]: vector<2xbf16>
gpu.func @lane_shuffle_pack_bf16(%a: vector<2xbf16>) -> vector<2xbf16> {
  // CHECK: %[[BITS:.*]] = llvm.bitcast %[[SRC]] : vector<2xbf16> to vector<2xi16>
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %[[BITS]] : (vector<2xi16>) -> i32
  // CHECK: llvm.bitcast %[[PACKED]] : i32 to vector<2xbf16>
  %0 = xegpu.lane_shuffle %a pack : vector<2xbf16>
  gpu.return %0 : vector<2xbf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_unpack_bf16
// CHECK-SAME:  %[[SRC:.*]]: vector<2xbf16>
gpu.func @lane_shuffle_unpack_bf16(%a: vector<2xbf16>) -> vector<2xbf16> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %[[SRC]] : vector<2xbf16> to i32
  // CHECK: %[[SHUF:.*]] = xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<2xi16>
  // CHECK: llvm.bitcast %[[SHUF]] : vector<2xi16> to vector<2xbf16>
  %0 = xegpu.lane_shuffle %a unpack : vector<2xbf16>
  gpu.return %0 : vector<2xbf16>
}
}

// -----

// The f8 element type is converted to a same-width integer by the type
// converter, so the fragment already is an integer vector by the time the
// shuffle is built and no further bitcast is needed on that side.
gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_unpack_f8
gpu.func @lane_shuffle_unpack_f8(%a: vector<4xf8E5M2>) -> vector<4xf8E5M2> {
  // CHECK: %[[PACKED:.*]] = llvm.bitcast %{{.*}} : vector<4xi8> to i32
  // CHECK: xevm.bitcast_shuffle %[[PACKED]] : (i32) -> vector<4xi8>
  %0 = xegpu.lane_shuffle %a unpack : vector<4xf8E5M2>
  gpu.return %0 : vector<4xf8E5M2>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: gpu.func @lane_shuffle_pack_i8
gpu.func @lane_shuffle_pack_i8(%a: vector<8xi8>) -> vector<8xi8> {
  // CHECK: %[[PACKED:.*]] = xevm.bitcast_shuffle %{{.*}} : (vector<8xi8>) -> i64
  // CHECK: llvm.bitcast %[[PACKED]] : i64 to vector<8xi8>
  %0 = xegpu.lane_shuffle %a pack : vector<8xi8>
  gpu.return %0 : vector<8xi8>
}
}
