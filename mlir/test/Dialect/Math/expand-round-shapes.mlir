// RUN: mlir-opt %s --math-expand-ops=ops=round | FileCheck %s
// RUN: mlir-opt %s --math-expand-ops=ops=tanh | FileCheck %s --check-prefix=FILTER

// CHECK-LABEL: func.func @round_fixed
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<4xi32>
// CHECK: math.copysign %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK: arith.select
// CHECK: return
// FILTER-LABEL: func.func @round_fixed
// FILTER: math.round
func.func @round_fixed(%x: vector<4xf32>) -> vector<4xf32> {
  %r = math.round %x : vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: func.func @round_scalable
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<[4]xi32>
// CHECK: math.copysign %{{.*}}, %{{.*}} : vector<[4]xf32>
// CHECK: arith.select
// CHECK: return
// FILTER-LABEL: func.func @round_scalable
// FILTER: math.round
func.func @round_scalable(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %r = math.round %x : vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// CHECK-LABEL: func.func @round_tensor
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<2x3xi32>
// CHECK: math.copysign %{{.*}}, %{{.*}} : tensor<2x3xf32>
// CHECK: arith.select
// CHECK: return
// FILTER-LABEL: func.func @round_tensor
// FILTER: math.round
func.func @round_tensor(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = math.round %x : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @round_f64
// CHECK: math.round
// FILTER-LABEL: func.func @round_f64
// FILTER: math.round
func.func @round_f64(%x: f64) -> f64 {
  %r = math.round %x : f64
  return %r : f64
}
