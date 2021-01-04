// RUN: mlir-opt -test-spirv-glsl-canonicalization -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: func @clamp_fordlessthan(%[[INPUT:.*]]: f32)
func @clamp_fordlessthan(%input: f32) -> f32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0.5 : f32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 1.0 : f32

  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThan %min, %input : f32
  %mid = spv.Select %0, %input, %min : i1, f32
  %1 = spv.FOrdLessThan %mid, %max : f32
  %2 = spv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK: func @clamp_fordlessthanequal(%[[INPUT:.*]]: f32)
func @clamp_fordlessthanequal(%input: f32) -> f32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0.5 : f32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 1.0 : f32

  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThanEqual %min, %input : f32
  %mid = spv.Select %0, %input, %min : i1, f32
  %1 = spv.FOrdLessThanEqual %mid, %max : f32
  %2 = spv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK: func @clamp_slessthan(%[[INPUT:.*]]: si32)
func @clamp_slessthan(%input: si32) -> si32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0 : si32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 10 : si32

  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThan %min, %input : si32
  %mid = spv.Select %0, %input, %min : i1, si32
  %1 = spv.SLessThan %mid, %max : si32
  %2 = spv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK: func @clamp_slessthanequal(%[[INPUT:.*]]: si32)
func @clamp_slessthanequal(%input: si32) -> si32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0 : si32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 10 : si32

  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThanEqual %min, %input : si32
  %mid = spv.Select %0, %input, %min : i1, si32
  %1 = spv.SLessThanEqual %mid, %max : si32
  %2 = spv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK: func @clamp_ulessthan(%[[INPUT:.*]]: i32)
func @clamp_ulessthan(%input: i32) -> i32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0 : i32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 10 : i32

  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThan %min, %input : i32
  %mid = spv.Select %0, %input, %min : i1, i32
  %1 = spv.ULessThan %mid, %max : i32
  %2 = spv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}

// -----

// CHECK: func @clamp_ulessthanequal(%[[INPUT:.*]]: i32)
func @clamp_ulessthanequal(%input: i32) -> i32 {
  // CHECK: %[[MIN:.*]] = spv.constant
  %min = spv.constant 0 : i32
  // CHECK: %[[MAX:.*]] = spv.constant
  %max = spv.constant 10 : i32

  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThanEqual %min, %input : i32
  %mid = spv.Select %0, %input, %min : i1, i32
  %1 = spv.ULessThanEqual %mid, %max : i32
  %2 = spv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}
