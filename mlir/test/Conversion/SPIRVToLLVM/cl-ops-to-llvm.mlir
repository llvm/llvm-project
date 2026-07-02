// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Unary floating-point OpenCL ops mapped to LLVM intrinsics
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cl_unary_float
spirv.func @cl_unary_float(%arg0: f32, %arg1: vector<3xf32>) "None" {
  // CHECK: llvm.intr.ceil(%{{.*}}) : (f32) -> f32
  %0 = spirv.CL.ceil %arg0 : f32
  // CHECK: llvm.intr.cos(%{{.*}}) : (f32) -> f32
  %1 = spirv.CL.cos %arg0 : f32
  // CHECK: llvm.intr.exp(%{{.*}}) : (f32) -> f32
  %2 = spirv.CL.exp %arg0 : f32
  // CHECK: llvm.intr.exp2(%{{.*}}) : (f32) -> f32
  %3 = spirv.CL.exp2 %arg0 : f32
  // CHECK: llvm.intr.exp10(%{{.*}}) : (f32) -> f32
  %4 = spirv.CL.exp10 %arg0 : f32
  // CHECK: llvm.intr.fabs(%{{.*}}) : (f32) -> f32
  %5 = spirv.CL.fabs %arg0 : f32
  // CHECK: llvm.intr.floor(%{{.*}}) : (f32) -> f32
  %6 = spirv.CL.floor %arg0 : f32
  // CHECK: llvm.intr.log(%{{.*}}) : (f32) -> f32
  %7 = spirv.CL.log %arg0 : f32
  // CHECK: llvm.intr.log2(%{{.*}}) : (f32) -> f32
  %8 = spirv.CL.log2 %arg0 : f32
  // CHECK: llvm.intr.log10(%{{.*}}) : (f32) -> f32
  %9 = spirv.CL.log10 %arg0 : f32
  // CHECK: llvm.intr.rint(%{{.*}}) : (f32) -> f32
  %10 = spirv.CL.rint %arg0 : f32
  // CHECK: llvm.intr.round(%{{.*}}) : (f32) -> f32
  %11 = spirv.CL.round %arg0 : f32
  // CHECK: llvm.intr.sin(%{{.*}}) : (f32) -> f32
  %12 = spirv.CL.sin %arg0 : f32
  // CHECK: llvm.intr.sinh(%{{.*}}) : (f32) -> f32
  %13 = spirv.CL.sinh %arg0 : f32
  // CHECK: llvm.intr.cosh(%{{.*}}) : (f32) -> f32
  %14 = spirv.CL.cosh %arg0 : f32
  // CHECK: llvm.intr.tan(%{{.*}}) : (f32) -> f32
  %15 = spirv.CL.tan %arg0 : f32
  // CHECK: llvm.intr.tanh(%{{.*}}) : (f32) -> f32
  %16 = spirv.CL.tanh %arg0 : f32
  // CHECK: llvm.intr.asin(%{{.*}}) : (f32) -> f32
  %17 = spirv.CL.asin %arg0 : f32
  // CHECK: llvm.intr.acos(%{{.*}}) : (f32) -> f32
  %18 = spirv.CL.acos %arg0 : f32
  // CHECK: llvm.intr.atan(%{{.*}}) : (f32) -> f32
  %19 = spirv.CL.atan %arg0 : f32
  // CHECK: llvm.intr.sqrt(%{{.*}}) : (f32) -> f32
  %20 = spirv.CL.sqrt %arg0 : f32
  // CHECK: llvm.intr.trunc(%{{.*}}) : (f32) -> f32
  %21 = spirv.CL.trunc %arg0 : f32
  // CHECK: llvm.intr.sin(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
  %22 = spirv.CL.sin %arg1 : vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// Binary floating-point OpenCL ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cl_binary_float
spirv.func @cl_binary_float(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.intr.pow(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %0 = spirv.CL.pow %arg0, %arg1 : f32
  // CHECK: llvm.intr.atan2(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %1 = spirv.CL.atan2 %arg0, %arg1 : f32
  // CHECK: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %2 = spirv.CL.fmax %arg0, %arg1 : f32
  // CHECK: llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  %3 = spirv.CL.fmin %arg0, %arg1 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// Ternary FP OpenCL ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cl_fma
spirv.func @cl_fma(%arg0: f32, %arg1: f32, %arg2: f32) "None" {
  // CHECK: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  %0 = spirv.CL.fma %arg0, %arg1, %arg2 : f32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// Integer OpenCL ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cl_integer
spirv.func @cl_integer(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.intr.smax(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %0 = spirv.CL.s_max %arg0, %arg1 : i32
  // CHECK: llvm.intr.smin(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %1 = spirv.CL.s_min %arg0, %arg1 : i32
  // CHECK: llvm.intr.umax(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %2 = spirv.CL.u_max %arg0, %arg1 : i32
  // CHECK: llvm.intr.umin(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %3 = spirv.CL.u_min %arg0, %arg1 : i32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.CL.mix
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mix_scalar
spirv.func @mix_scalar(%x: f32, %y: f32, %a: f32) "None" {
  // CHECK: %[[DIFF:.*]] = llvm.fsub %{{.*}}, %{{.*}} : f32
  // CHECK: llvm.intr.fma(%{{.*}}, %[[DIFF]], %{{.*}}) : (f32, f32, f32) -> f32
  %0 = spirv.CL.mix %x, %y, %a : f32
  spirv.Return
}

// CHECK-LABEL: @mix_vector
spirv.func @mix_vector(%x: vector<4xf32>, %y: vector<4xf32>, %a: vector<4xf32>) "None" {
  // CHECK: %[[DIFF:.*]] = llvm.fsub %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: llvm.intr.fma(%{{.*}}, %[[DIFF]], %{{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %0 = spirv.CL.mix %x, %y, %a : vector<4xf32>
  spirv.Return
}
