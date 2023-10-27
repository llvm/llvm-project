// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.GL.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ceil
spirv.func @ceil(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.ceil(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Ceil %arg0 : f32
  // CHECK: llvm.intr.ceil(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Ceil %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cos
spirv.func @cos(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.cos(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Cos %arg0 : f32
  // CHECK: llvm.intr.cos(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Cos %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @exp
spirv.func @exp(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.exp(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Exp %arg0 : f32
  // CHECK: llvm.intr.exp(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Exp %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.FAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fabs
spirv.func @fabs(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.fabs(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.FAbs %arg0 : f32
  // CHECK: llvm.intr.fabs(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.FAbs %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @floor
spirv.func @floor(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.floor(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Floor %arg0 : f32
  // CHECK: llvm.intr.floor(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Floor %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.FMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmax
spirv.func @fmax(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.FMax %arg0, %arg0 : f32
  // CHECK: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.FMax %arg1, %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.FMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmin
spirv.func @fmin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.FMin %arg0, %arg0 : f32
  // CHECK: llvm.intr.minnum(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.FMin %arg1, %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @log
spirv.func @log(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.log(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Log %arg0 : f32
  // CHECK: llvm.intr.log(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Log %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sin
spirv.func @sin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.sin(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Sin %arg0 : f32
  // CHECK: llvm.intr.sin(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Sin %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.SMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smax
spirv.func @smax(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.intr.smax(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spirv.GL.SMax %arg0, %arg0 : i16
  // CHECK: llvm.intr.smax(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spirv.GL.SMax %arg1, %arg1 : vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.SMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smin
spirv.func @smin(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.intr.smin(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spirv.GL.SMin %arg0, %arg0 : i16
  // CHECK: llvm.intr.smin(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spirv.GL.SMin %arg1, %arg1 : vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Sqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sqrt
spirv.func @sqrt(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.sqrt(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Sqrt %arg0 : f32
  // CHECK: llvm.intr.sqrt(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Sqrt %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Tan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tan
spirv.func @tan(%arg0: f32) "None" {
  // CHECK: %[[SIN:.*]] = llvm.intr.sin(%{{.*}}) : (f32) -> f32
  // CHECK: %[[COS:.*]] = llvm.intr.cos(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[SIN]], %[[COS]] : f32
  %0 = spirv.GL.Tan %arg0 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tanh
spirv.func @tanh(%arg0: f32) "None" {
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: %[[X2:.*]] = llvm.fmul %[[TWO]], %{{.*}} : f32
  // CHECK: %[[EXP:.*]] = llvm.intr.exp(%[[X2]]) : (f32) -> f32
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[T0:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : f32
  // CHECK: %[[T1:.*]] = llvm.fadd %[[EXP]], %[[ONE]] : f32
  // CHECK: llvm.fdiv %[[T0]], %[[T1]] : f32
  %0 = spirv.GL.Tanh %arg0 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.InverseSqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @inverse_sqrt
spirv.func @inverse_sqrt(%arg0: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fdiv %[[ONE]], %[[SQRT]] : f32
  %0 = spirv.GL.InverseSqrt %arg0 : f32
  spirv.Return
}
