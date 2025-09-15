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

    return
  }

}
