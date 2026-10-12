// RUN: mlir-opt -canonicalize %s | FileCheck %s

// An f8 (E4M3) value survives the round trip through f16, so the pair folds
// away and the packed source is returned directly.
// CHECK-LABEL: @fold_f8_round_trip
// CHECK-SAME:    (%[[ARG0:.*]]: vector<8xi8>)
// CHECK-NOT:     xevm.extf
// CHECK-NOT:     xevm.truncf
// CHECK:         return %[[ARG0]] : vector<8xi8>
func.func @fold_f8_round_trip(%arg0: vector<8xi8>) -> vector<8xi8> {
  %ext = xevm.extf %arg0 { src_etype=f8, dst_etype=f16 } : (vector<8xi8>) -> vector<8xf16>
  %trunc = xevm.truncf %ext { src_etype=f16, dst_etype=f8 } : (vector<8xf16>) -> vector<8xi8>
  return %trunc : vector<8xi8>
}

// e2m1 has no infinities and no NaNs, so it round trips through bf16 as well.
// CHECK-LABEL: @fold_e2m1_round_trip
// CHECK-SAME:    (%[[ARG0:.*]]: vector<8xi4>)
// CHECK-NOT:     xevm.extf
// CHECK-NOT:     xevm.truncf
// CHECK:         return %[[ARG0]] : vector<8xi4>
func.func @fold_e2m1_round_trip(%arg0: vector<8xi4>) -> vector<8xi4> {
  %ext = xevm.extf %arg0 { src_etype=e2m1, dst_etype=bf16 } : (vector<8xi4>) -> vector<8xbf16>
  %trunc = xevm.truncf %ext { src_etype=bf16, dst_etype=e2m1 } : (vector<8xbf16>) -> vector<8xi4>
  return %trunc : vector<8xi4>
}

// CHECK-LABEL: @fold_f8_round_trip_scalar
// CHECK-SAME:    (%[[ARG0:.*]]: i8)
// CHECK-NOT:     xevm.extf
// CHECK-NOT:     xevm.truncf
// CHECK:         return %[[ARG0]] : i8
func.func @fold_f8_round_trip_scalar(%arg0: i8) -> i8 {
  %ext = xevm.extf %arg0 { src_etype=f8, dst_etype=f16 } : (i8) -> f16
  %trunc = xevm.truncf %ext { src_etype=f16, dst_etype=f8 } : (f16) -> i8
  return %trunc : i8
}

// bf8 is IEEE-like: extending a signaling NaN quiets it, so the narrow value is
// not recoverable and the round trip must be kept.
// CHECK-LABEL: @no_fold_bf8_round_trip
// CHECK:         %[[EXT:.*]] = xevm.extf %{{.*}} {src_etype = bf8, dst_etype = f16}
// CHECK:         %[[TRUNC:.*]] = xevm.truncf %[[EXT]] {src_etype = f16, dst_etype = bf8}
// CHECK:         return %[[TRUNC]]
func.func @no_fold_bf8_round_trip(%arg0: vector<8xi8>) -> vector<8xi8> {
  %ext = xevm.extf %arg0 { src_etype=bf8, dst_etype=f16 } : (vector<8xi8>) -> vector<8xf16>
  %trunc = xevm.truncf %ext { src_etype=f16, dst_etype=bf8 } : (vector<8xf16>) -> vector<8xi8>
  return %trunc : vector<8xi8>
}

// The narrow formats of the two ops differ, so the pair is not a round trip.
// CHECK-LABEL: @no_fold_narrow_format_mismatch
// CHECK:         %[[EXT:.*]] = xevm.extf %{{.*}} {src_etype = f8, dst_etype = f16}
// CHECK:         %[[TRUNC:.*]] = xevm.truncf %[[EXT]] {src_etype = f16, dst_etype = bf8}
// CHECK:         return %[[TRUNC]]
func.func @no_fold_narrow_format_mismatch(%arg0: vector<8xi8>) -> vector<8xi8> {
  %ext = xevm.extf %arg0 { src_etype=f8, dst_etype=f16 } : (vector<8xi8>) -> vector<8xf16>
  %trunc = xevm.truncf %ext { src_etype=f16, dst_etype=bf8 } : (vector<8xf16>) -> vector<8xi8>
  return %trunc : vector<8xi8>
}

// The wide formats of the two ops differ: the value is extended as f16 and
// truncated as if it were bf16.
// CHECK-LABEL: @no_fold_wide_format_mismatch
// CHECK:         %[[EXT:.*]] = xevm.extf %{{.*}} {src_etype = f8, dst_etype = f16}
// CHECK:         %[[TRUNC:.*]] = xevm.truncf %[[EXT]] {src_etype = bf16, dst_etype = f8}
// CHECK:         return %[[TRUNC]]
func.func @no_fold_wide_format_mismatch(%arg0: vector<8xi8>) -> vector<8xi8> {
  %ext = xevm.extf %arg0 { src_etype=f8, dst_etype=f16 } : (vector<8xi8>) -> vector<8xf16>
  %trunc = xevm.truncf %ext { src_etype=bf16, dst_etype=f8 } : (vector<8xf16>) -> vector<8xi8>
  return %trunc : vector<8xi8>
}

// The other order is lossy: the truncation to f8 cannot be undone.
// CHECK-LABEL: @no_fold_lossy_round_trip
// CHECK:         %[[TRUNC:.*]] = xevm.truncf %{{.*}} {src_etype = f16, dst_etype = f8}
// CHECK:         %[[EXT:.*]] = xevm.extf %[[TRUNC]] {src_etype = f8, dst_etype = f16}
// CHECK:         return %[[EXT]]
func.func @no_fold_lossy_round_trip(%arg0: vector<8xf16>) -> vector<8xf16> {
  %trunc = xevm.truncf %arg0 { src_etype=f16, dst_etype=f8 } : (vector<8xf16>) -> vector<8xi8>
  %ext = xevm.extf %trunc { src_etype=f8, dst_etype=f16 } : (vector<8xi8>) -> vector<8xf16>
  return %ext : vector<8xf16>
}
