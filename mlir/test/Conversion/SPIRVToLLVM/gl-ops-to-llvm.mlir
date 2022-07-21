// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GL.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ceil
spv.func @ceil(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Ceil %arg0 : f32
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Ceil %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cos
spv.func @cos(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Cos %arg0 : f32
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Cos %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @exp
spv.func @exp(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Exp %arg0 : f32
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Exp %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.FAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fabs
spv.func @fabs(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.FAbs %arg0 : f32
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.FAbs %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @floor
spv.func @floor(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Floor %arg0 : f32
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Floor %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.FMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmax
spv.func @fmax(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spv.GL.FMax %arg0, %arg0 : f32
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.FMax %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.FMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmin
spv.func @fmin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spv.GL.FMin %arg0, %arg0 : f32
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.FMin %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @log
spv.func @log(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.log"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Log %arg0 : f32
  // CHECK: "llvm.intr.log"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Log %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sin
spv.func @sin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Sin %arg0 : f32
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Sin %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.SMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smax
spv.func @smax(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spv.GL.SMax %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spv.GL.SMax %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.SMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smin
spv.func @smin(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spv.GL.SMin %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spv.GL.SMin %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Sqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sqrt
spv.func @sqrt(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  %0 = spv.GL.Sqrt %arg0 : f32
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spv.GL.Sqrt %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Tan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tan
spv.func @tan(%arg0: f32) "None" {
  // CHECK: %[[SIN:.*]] = "llvm.intr.sin"(%{{.*}}) : (f32) -> f32
  // CHECK: %[[COS:.*]] = "llvm.intr.cos"(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[SIN]], %[[COS]] : f32
  %0 = spv.GL.Tan %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.Tanh
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
  %0 = spv.GL.Tanh %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GL.InverseSqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @inverse_sqrt
spv.func @inverse_sqrt(%arg0: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[ONE]], %[[SQRT]] : f32
  %0 = spv.GL.InverseSqrt %arg0 : f32
  spv.Return
}
