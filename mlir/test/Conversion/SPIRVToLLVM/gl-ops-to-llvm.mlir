// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

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
// spirv.GL.FClamp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fclamp
spirv.func @fclamp(%arg0: f32, %arg1: f32, %arg2: f32) "None" {
  // CHECK: %[[MAX:.*]] = llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // CHECK: llvm.intr.minnum(%[[MAX]], %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.FClamp %arg0, %arg1, %arg2 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.NMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @nmax
spirv.func @nmax(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.NMax %arg0, %arg0 : f32
  // CHECK: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.NMax %arg1, %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.NMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @nmin
spirv.func @nmin(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.NMin %arg0, %arg0 : f32
  // CHECK: llvm.intr.minnum(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.NMin %arg1, %arg1 : vector<3xf16>
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
// spirv.GL.Pow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @pow
spirv.func @pow(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.pow(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.GL.Pow %arg0, %arg0 : f32
  // CHECK: llvm.intr.pow(%{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Pow %arg1, %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Fma
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fma
spirv.func @fma(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  %0 = spirv.GL.Fma %arg0, %arg0, %arg0 : f32
  // CHECK: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<3xf16>, vector<3xf16>, vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Fma %arg1, %arg1, %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.SAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sabs
spirv.func @sabs(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (i16) -> i16
  %0 = spirv.GL.SAbs %arg0 : i16
  // CHECK: "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (vector<3xi32>) -> vector<3xi32>
  %1 = spirv.GL.SAbs %arg1 : vector<3xi32>
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
// spirv.GL.UMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umax
spirv.func @umax(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.intr.umax(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spirv.GL.UMax %arg0, %arg0 : i16
  // CHECK: llvm.intr.umax(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spirv.GL.UMax %arg1, %arg1 : vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.UMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umin
spirv.func @umin(%arg0: i16, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.intr.umin(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  %0 = spirv.GL.UMin %arg0, %arg0 : i16
  // CHECK: llvm.intr.umin(%{{.*}}, %{{.*}}) : (vector<3xi32>, vector<3xi32>) -> vector<3xi32>
  %1 = spirv.GL.UMin %arg1, %arg1 : vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.SClamp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sclamp
spirv.func @sclamp(%arg0: i16, %arg1: i16, %arg2: i16) "None" {
  // CHECK: %[[MAX:.*]] = llvm.intr.smax(%{{.*}}, %{{.*}}) : (i16, i16) -> i16
  // CHECK: llvm.intr.smin(%[[MAX]], %{{.*}}) : (i16, i16) -> i16
  %0 = spirv.GL.SClamp %arg0, %arg1, %arg2 : i16
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.UClamp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @uclamp
spirv.func @uclamp(%arg0: i32, %arg1: i32, %arg2: i32) "None" {
  // CHECK: %[[MAX:.*]] = llvm.intr.umax(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  // CHECK: llvm.intr.umin(%[[MAX]], %{{.*}}) : (i32, i32) -> i32
  %0 = spirv.GL.UClamp %arg0, %arg1, %arg2 : i32
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
  // CHECK: llvm.intr.tan(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Tan %arg0 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tanh
spirv.func @tanh(%arg0: f32) "None" {
  // CHECK: llvm.intr.tanh(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Tanh %arg0 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Exp2
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @exp2
spirv.func @exp2(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.exp2(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Exp2 %arg0 : f32
  // CHECK: llvm.intr.exp2(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Exp2 %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Log2
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @log2
spirv.func @log2(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.log2(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Log2 %arg0 : f32
  // CHECK: llvm.intr.log2(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Log2 %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Round
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @round
spirv.func @round(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.round(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Round %arg0 : f32
  // CHECK: llvm.intr.round(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Round %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.RoundEven
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @round_even
spirv.func @round_even(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.roundeven(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.RoundEven %arg0 : f32
  // CHECK: llvm.intr.roundeven(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.RoundEven %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Sinh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sinh
spirv.func @sinh(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.sinh(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Sinh %arg0 : f32
  // CHECK: llvm.intr.sinh(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Sinh %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Cosh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cosh
spirv.func @cosh(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.cosh(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Cosh %arg0 : f32
  // CHECK: llvm.intr.cosh(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Cosh %arg1 : vector<3xf16>
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

//===----------------------------------------------------------------------===//
// spirv.GL.Radians
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @radians
spirv.func @radians(%arg0: f32, %arg1: vector<3xf32>) "None" {
  // CHECK: %[[FACTOR:.*]] = llvm.mlir.constant(0.0174532924 : f32) : f32
  // CHECK: llvm.fmul %{{.*}}, %[[FACTOR]] : f32
  %0 = spirv.GL.Radians %arg0 : f32
  // CHECK: %[[VFACTOR:.*]] = llvm.mlir.constant(dense<0.0174532924> : vector<3xf32>) : vector<3xf32>
  // CHECK: llvm.fmul %{{.*}}, %[[VFACTOR]] : vector<3xf32>
  %1 = spirv.GL.Radians %arg1 : vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Degrees
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @degrees
spirv.func @degrees(%arg0: f32, %arg1: vector<3xf32>) "None" {
  // CHECK: %[[FACTOR:.*]] = llvm.mlir.constant(57.2957802 : f32) : f32
  // CHECK: llvm.fmul %{{.*}}, %[[FACTOR]] : f32
  %0 = spirv.GL.Degrees %arg0 : f32
  // CHECK: %[[VFACTOR:.*]] = llvm.mlir.constant(dense<57.2957802> : vector<3xf32>) : vector<3xf32>
  // CHECK: llvm.fmul %{{.*}}, %[[VFACTOR]] : vector<3xf32>
  %1 = spirv.GL.Degrees %arg1 : vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Trunc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @trunc
spirv.func @trunc(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.trunc(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Trunc %arg0 : f32
  // CHECK: llvm.intr.trunc(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Trunc %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Asin, spirv.GL.Acos, spirv.GL.Atan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @asin_acos_atan
spirv.func @asin_acos_atan(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: llvm.intr.asin(%{{.*}}) : (f32) -> f32
  %0 = spirv.GL.Asin %arg0 : f32
  // CHECK: llvm.intr.acos(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  %1 = spirv.GL.Acos %arg1 : vector<3xf16>
  // CHECK: llvm.intr.atan(%{{.*}}) : (f32) -> f32
  %2 = spirv.GL.Atan %arg0 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.Fract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fract
spirv.func @fract(%arg0: f32, %arg1: vector<3xf16>) "None" {
  // CHECK: %[[FLOOR:.*]] = llvm.intr.floor(%{{.*}}) : (f32) -> f32
  // CHECK: llvm.fsub %{{.*}}, %[[FLOOR]] : f32
  %0 = spirv.GL.Fract %arg0 : f32
  // CHECK: %[[FLOORV:.*]] = llvm.intr.floor(%{{.*}}) : (vector<3xf16>) -> vector<3xf16>
  // CHECK: llvm.fsub %{{.*}}, %[[FLOORV]] : vector<3xf16>
  %1 = spirv.GL.Fract %arg1 : vector<3xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GL.FMix
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmix_scalar
spirv.func @fmix_scalar(%x: f32, %y: f32, %a: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[ONE_MINUS_A:.*]] = llvm.fsub %[[ONE]], %{{.*}} : f32
  // CHECK: %[[LHS:.*]] = llvm.fmul %{{.*}}, %[[ONE_MINUS_A]] : f32
  // CHECK: %[[RHS:.*]] = llvm.fmul %{{.*}}, %{{.*}} : f32
  // CHECK: llvm.fadd %[[LHS]], %[[RHS]] : f32
  %0 = spirv.GL.FMix %x : f32, %y : f32, %a : f32 -> f32
  spirv.Return
}

// CHECK-LABEL: @fmix_vector
spirv.func @fmix_vector(%x: vector<4xf32>, %y: vector<4xf32>, %a: vector<4xf32>) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[ONE_MINUS_A:.*]] = llvm.fsub %[[ONE]], %{{.*}} : vector<4xf32>
  // CHECK: %[[LHS:.*]] = llvm.fmul %{{.*}}, %[[ONE_MINUS_A]] : vector<4xf32>
  // CHECK: %[[RHS:.*]] = llvm.fmul %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: llvm.fadd %[[LHS]], %[[RHS]] : vector<4xf32>
  %0 = spirv.GL.FMix %x : vector<4xf32>, %y : vector<4xf32>, %a : vector<4xf32> -> vector<4xf32>
  spirv.Return
}
