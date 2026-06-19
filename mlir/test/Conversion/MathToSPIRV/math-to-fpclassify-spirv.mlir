// RUN: mlir-opt --convert-math-to-spirv %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

  // CHECK-LABEL: @fpclassify
  func.func @fpclassify(%x: f32, %v: vector<4xf32>) {
    // CHECK: spirv.IsFinite %{{.*}} : f32
    %0 = math.isfinite %x : f32
    // CHECK: spirv.IsFinite %{{.*}} : vector<4xf32>
    %1 = math.isfinite %v : vector<4xf32>

    // CHECK: spirv.IsNan %{{.*}} : f32
    %2 = math.isnan %x : f32
    // CHECK: spirv.IsNan %{{.*}} : vector<4xf32>
    %3 = math.isnan %v : vector<4xf32>

    // CHECK: spirv.IsInf %{{.*}} : f32
    %4 = math.isinf %x : f32
    // CHECK: spirv.IsInf %{{.*}} : vector<4xf32>
    %5 = math.isinf %v : vector<4xf32>

    // CHECK: spirv.IsNormal %{{.*}} : f32
    %6 = math.isnormal %x : f32
    // CHECK: spirv.IsNormal %{{.*}} : vector<4xf32>
    %7 = math.isnormal %v : vector<4xf32>

    return
  }

  // CHECK-LABEL: @ctpop
  func.func @ctpop(%i: i32, %iv: vector<4xi32>) {
    // CHECK: spirv.BitCount %{{.*}} : i32
    %0 = math.ctpop %i : i32
    // CHECK: spirv.BitCount %{{.*}} : vector<4xi32>
    %1 = math.ctpop %iv : vector<4xi32>
    return
  }

}
