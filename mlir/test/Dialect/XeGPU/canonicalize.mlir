// RUN: mlir-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fold_lane_shuffle_pack_unpack
// CHECK-SAME: %[[ARG0:.+]]: vector<2xi16>
// CHECK-NOT: xegpu.lane_shuffle
// CHECK: return %[[ARG0]]
func.func @fold_lane_shuffle_pack_unpack(%a: vector<2xi16>) -> vector<2xi16> {
  %0 = xegpu.lane_shuffle %a pack : vector<2xi16>
  %1 = xegpu.lane_shuffle %0 unpack : vector<2xi16>
  return %1 : vector<2xi16>
}

// -----

// CHECK-LABEL: func.func @fold_lane_shuffle_unpack_pack
// CHECK-SAME: %[[ARG0:.+]]: vector<4xf8E5M2>
// CHECK-NOT: xegpu.lane_shuffle
// CHECK: return %[[ARG0]]
func.func @fold_lane_shuffle_unpack_pack(%a: vector<4xf8E5M2>) -> vector<4xf8E5M2> {
  %0 = xegpu.lane_shuffle %a unpack : vector<4xf8E5M2>
  %1 = xegpu.lane_shuffle %0 pack : vector<4xf8E5M2>
  return %1 : vector<4xf8E5M2>
}

// -----

// Two shuffles in the same direction are not inverses and must not fold.

// CHECK-LABEL: func.func @no_fold_lane_shuffle_pack_pack
// CHECK: xegpu.lane_shuffle {{.*}} pack
// CHECK: xegpu.lane_shuffle {{.*}} pack
func.func @no_fold_lane_shuffle_pack_pack(%a: vector<2xi16>) -> vector<2xi16> {
  %0 = xegpu.lane_shuffle %a pack : vector<2xi16>
  %1 = xegpu.lane_shuffle %0 pack : vector<2xi16>
  return %1 : vector<2xi16>
}
