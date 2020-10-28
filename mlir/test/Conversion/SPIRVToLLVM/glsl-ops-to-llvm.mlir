// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GLSL.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ceil
spv.func @ceil(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Ceil %arg0 : f32
  // CHECK: "llvm.intr.ceil"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Ceil %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cos
spv.func @cos(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Cos %arg0 : f32
  // CHECK: "llvm.intr.cos"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Cos %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @exp
spv.func @exp(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Exp %arg0 : f32
  // CHECK: "llvm.intr.exp"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Exp %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fabs
spv.func @fabs(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.FAbs %arg0 : f32
  // CHECK: "llvm.intr.fabs"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.FAbs %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @floor
spv.func @floor(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Floor %arg0 : f32
  // CHECK: "llvm.intr.floor"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Floor %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmax
spv.func @fmax(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (!llvm.float, !llvm.float) -> !llvm.float
  %0 = spv.GLSL.FMax %arg0, %arg0 : f32
  // CHECK: "llvm.intr.maxnum"(%{{.*}}, %{{.*}}) : (!llvm.vec<3 x half>, !llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.FMax %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmin
spv.func @fmin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (!llvm.float, !llvm.float) -> !llvm.float
  %0 = spv.GLSL.FMin %arg0, %arg0 : f32
  // CHECK: "llvm.intr.minnum"(%{{.*}}, %{{.*}}) : (!llvm.vec<3 x half>, !llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.FMin %arg1, %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @log
spv.func @log(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.log"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Log %arg0 : f32
  // CHECK: "llvm.intr.log"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Log %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sin
spv.func @sin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Sin %arg0 : f32
  // CHECK: "llvm.intr.sin"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Sin %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.SMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smax
spv.func @smax(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (!llvm.i16, !llvm.i16) -> !llvm.i16
  %0 = spv.GLSL.SMax %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smax"(%{{.*}}, %{{.*}}) : (!llvm.vec<3 x i32>, !llvm.vec<3 x i32>) -> !llvm.vec<3 x i32>
  %1 = spv.GLSL.SMax %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.SMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smin
spv.func @smin(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (!llvm.i16, !llvm.i16) -> !llvm.i16
  %0 = spv.GLSL.SMin %arg0, %arg0 : i16
  // CHECK: "llvm.intr.smin"(%{{.*}}, %{{.*}}) : (!llvm.vec<3 x i32>, !llvm.vec<3 x i32>) -> !llvm.vec<3 x i32>
  %1 = spv.GLSL.SMin %arg1, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sqrt
spv.func @sqrt(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (!llvm.float) -> !llvm.float
  %0 = spv.GLSL.Sqrt %arg0 : f32
  // CHECK: "llvm.intr.sqrt"(%{{.*}}) : (!llvm.vec<3 x half>) -> !llvm.vec<3 x half>
  %1 = spv.GLSL.Sqrt %arg1 : vector<3xf16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tan
spv.func @tan(%arg0: f32) "None" {
  // CHECK: %[[SIN:.*]] = "llvm.intr.sin"(%{{.*}}) : (!llvm.float) -> !llvm.float
  // CHECK: %[[COS:.*]] = "llvm.intr.cos"(%{{.*}}) : (!llvm.float) -> !llvm.float
  // CHECK: llvm.fdiv %[[SIN]], %[[COS]] : !llvm.float
  %0 = spv.GLSL.Tan %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tanh
spv.func @tanh(%arg0: f32) "None" {
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
  // CHECK: %[[X2:.*]] = llvm.fmul %[[TWO]], %{{.*}} : !llvm.float
  // CHECK: %[[EXP:.*]] = "llvm.intr.exp"(%[[X2]]) : (!llvm.float) -> !llvm.float
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
  // CHECK: %[[T0:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : !llvm.float
  // CHECK: %[[T1:.*]] = llvm.fadd %[[EXP]], %[[ONE]] : !llvm.float
  // CHECK: llvm.fdiv %[[T0]], %[[T1]] : !llvm.float
  %0 = spv.GLSL.Tanh %arg0 : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.InverseSqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @inverse_sqrt
spv.func @inverse_sqrt(%arg0: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%{{.*}}) : (!llvm.float) -> !llvm.float
  // CHECK: llvm.fdiv %[[ONE]], %[[SQRT]] : !llvm.float
  %0 = spv.GLSL.InverseSqrt %arg0 : f32
  spv.Return
}
