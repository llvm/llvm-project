// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GLSL.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ceil
spv.func @ceil(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Ceil %arg0 : f32
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Ceil %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cos
spv.func @cos(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Cos %arg0 : f32
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Cos %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @exp
spv.func @exp(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Exp %arg0 : f32
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Exp %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fabs
spv.func @fabs(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.FAbs %arg0 : f32
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.FAbs %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @floor
spv.func @floor(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Floor %arg0 : f32
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Floor %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmax
spv.func @fmax(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spv.GLSL.FMax %arg0, %arg0 : f32
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.FMax %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmin
spv.func @fmin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spv.GLSL.FMin %arg0, %arg0 : f32
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.FMin %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @log
spv.func @log(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.log"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Log %arg0 : f32
  // CHECK: "llvm.intr.log"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Log %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sin
spv.func @sin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Sin %arg0 : f32
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Sin %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.SMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smax
spv.func @smax(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spv.GLSL.SMax %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spv.GLSL.SMax %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.SMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smin
spv.func @smin(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spv.GLSL.SMin %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spv.GLSL.SMin %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sqrt
spv.func @sqrt(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  %0 = spv.GLSL.Sqrt %arg0 : f32
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GLSL.Sqrt %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tan
spv.func @tan(%arg0: f32) "None" {
  // CHECK: %[[SIN:.*]] = "llvm.intr.sin"(%{{.*}}) : (f32) -> f32
  // CHECK: %[[COS:.*]] = "llvm.intr.cos"(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[SIN]], %[[COS]] : f32
  %0 = spv.GLSL.Tan %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tanh
spv.func @tanh(%arg0: f32) "None" {
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: %[[X2:.*]] = llvm.fmul %[[TWO]], %{{.*}} : f32
  // CHECK: %[[EXP:.*]] = "llvm.intr.exp"(%[[X2]]) : (f32) -> f32
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[T0:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : f32
  // CHECK: %[[T1:.*]] = llvm.fadd %[[EXP]], %[[ONE]] : f32
  // CHECK: llvm.fdiv %[[T0]], %[[T1]] : f32
  %0 = spv.GLSL.Tanh %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.InverseSqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @inverse_sqrt
spv.func @inverse_sqrt(%arg0: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[ONE]], %[[SQRT]] : f32
  %0 = spv.GLSL.InverseSqrt %arg0 : f32
  spv.Return
}
