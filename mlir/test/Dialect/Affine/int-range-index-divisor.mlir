// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// The divisor is 1 at index width 32. Neither division can be replaced with
// poison solely because the divisor is negative at the storage width.

// CHECK-LABEL: func @affine_floor_divisor
// CHECK: %[[DIV:.*]] = affine.apply
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[DIV]],
// CHECK: return %[[SELECT]]
func.func @affine_floor_divisor(%condition: i1, %x: index) -> index {
  %divisor = arith.constant -4294967295 : index
  %seven = arith.constant 7 : index
  %result = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%x)[%divisor]
  %selected = arith.select %condition, %result, %seven : index
  return %selected : index
}

// CHECK-LABEL: func @affine_ceil_divisor
// CHECK: %[[DIV:.*]] = affine.apply
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[DIV]],
// CHECK: return %[[SELECT]]
func.func @affine_ceil_divisor(%condition: i1, %x: index) -> index {
  %divisor = arith.constant -4294967295 : index
  %seven = arith.constant 7 : index
  %result = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)>(%x)[%divisor]
  %selected = arith.select %condition, %result, %seven : index
  return %selected : index
}
