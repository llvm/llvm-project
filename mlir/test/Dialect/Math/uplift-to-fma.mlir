// RUN: mlir-opt %s --split-input-file --math-uplift-to-fma | FileCheck %s

// No uplifting without fastmath flags.
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
//       CHECK: %[[V1:.*]] = arith.mulf %[[ARG1]], %[[ARG2]]
//       CHECK: %[[V2:.*]] = arith.addf %[[V1]], %[[ARG3]]
//       CHECK: return %[[V2]]
func.func @test(%arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
  %1 = arith.mulf %arg1, %arg2 : f32
  %2 = arith.addf %1, %arg3 : f32
  return %2 : f32
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
//       CHECK: %[[RES:.*]] = math.fma %[[ARG1]], %[[ARG2]], %[[ARG3]] fastmath<fast> : f32
//       CHECK: return %[[RES]]
func.func @test(%arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
  %1 = arith.mulf %arg1, %arg2 fastmath<fast> : f32
  %2 = arith.addf %1, %arg3 fastmath<fast> : f32
  return %2 : f32
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
//       CHECK: %[[RES:.*]] = math.fma %[[ARG1]], %[[ARG2]], %[[ARG3]] fastmath<contract> : f32
//       CHECK: return %[[RES]]
func.func @test(%arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
  %1 = arith.mulf %arg1, %arg2 fastmath<fast> : f32
  %2 = arith.addf %arg3, %1 fastmath<contract> : f32
  return %2 : f32
}
