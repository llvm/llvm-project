// RUN: mlir-opt -split-input-file -spirv-canonicalize-gl %s | FileCheck %s

// CHECK-LABEL: func @clamp_fordlessthan
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func.func @clamp_fordlessthan(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spirv.GL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.FOrdLessThan %min, %input : f32
  %mid = spirv.Select %0, %input, %min : i1, f32
  %1 = spirv.FOrdLessThan %mid, %max : f32
  %2 = spirv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthan
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func.func @clamp_fordlessthan(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spirv.GL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.FOrdLessThan %input, %min : f32
  %mid = spirv.Select %0, %min, %input : i1, f32
  %1 = spirv.FOrdLessThan %max, %input : f32
  %2 = spirv.Select %1, %max, %mid : i1, f32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func.func @clamp_fordlessthanequal(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spirv.GL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.FOrdLessThanEqual %min, %input : f32
  %mid = spirv.Select %0, %input, %min : i1, f32
  %1 = spirv.FOrdLessThanEqual %mid, %max : f32
  %2 = spirv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func.func @clamp_fordlessthanequal(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spirv.GL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.FOrdLessThanEqual %input, %min : f32
  %mid = spirv.Select %0, %min, %input : i1, f32
  %1 = spirv.FOrdLessThanEqual %max, %input : f32
  %2 = spirv.Select %1, %max, %mid : i1, f32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_slessthan
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func.func @clamp_slessthan(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spirv.GL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.SLessThan %min, %input : si32
  %mid = spirv.Select %0, %input, %min : i1, si32
  %1 = spirv.SLessThan %mid, %max : si32
  %2 = spirv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthan
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func.func @clamp_slessthan(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spirv.GL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.SLessThan %input, %min : si32
  %mid = spirv.Select %0, %min, %input : i1, si32
  %1 = spirv.SLessThan %max, %input : si32
  %2 = spirv.Select %1, %max, %mid : i1, si32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func.func @clamp_slessthanequal(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spirv.GL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.SLessThanEqual %min, %input : si32
  %mid = spirv.Select %0, %input, %min : i1, si32
  %1 = spirv.SLessThanEqual %mid, %max : si32
  %2 = spirv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func.func @clamp_slessthanequal(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spirv.GL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.SLessThanEqual %input, %min : si32
  %mid = spirv.Select %0, %min, %input : i1, si32
  %1 = spirv.SLessThanEqual %max, %input : si32
  %2 = spirv.Select %1, %max, %mid : i1, si32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_ulessthan
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func.func @clamp_ulessthan(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spirv.GL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.ULessThan %min, %input : i32
  %mid = spirv.Select %0, %input, %min : i1, i32
  %1 = spirv.ULessThan %mid, %max : i32
  %2 = spirv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthan
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func.func @clamp_ulessthan(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spirv.GL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.ULessThan %input, %min : i32
  %mid = spirv.Select %0, %min, %input : i1, i32
  %1 = spirv.ULessThan %max, %input : i32
  %2 = spirv.Select %1, %max, %mid : i1, i32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func.func @clamp_ulessthanequal(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spirv.GL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.ULessThanEqual %min, %input : i32
  %mid = spirv.Select %0, %input, %min : i1, i32
  %1 = spirv.ULessThanEqual %mid, %max : i32
  %2 = spirv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func.func @clamp_ulessthanequal(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spirv.GL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spirv.ULessThanEqual %input, %min : i32
  %mid = spirv.Select %0, %min, %input : i1, i32
  %1 = spirv.ULessThanEqual %max, %input : i32
  %2 = spirv.Select %1, %max, %mid : i1, i32

  // CHECK-NEXT: spirv.ReturnValue [[RES]]
  spirv.ReturnValue %2 : i32
}
