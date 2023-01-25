// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @iequal_scalar(%arg0: i32, %arg1: i32)  "None" {
    // CHECK: {{.*}} = spirv.IEqual {{.*}}, {{.*}} : i32
    %0 = spirv.IEqual %arg0, %arg1 : i32
    spirv.Return
  }
  spirv.func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.INotEqual %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.SGreaterThan %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.SLessThan %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.UGreaterThan %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spirv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.ULessThan %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>)  "None" {
    // CHECK: {{.*}} = spirv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spirv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @cmpf(%arg0 : f32, %arg1 : f32) "None" {
    // CHECK: spirv.FOrdEqual
    %1 = spirv.FOrdEqual %arg0, %arg1 : f32
    // CHECK: spirv.FOrdGreaterThan
    %2 = spirv.FOrdGreaterThan %arg0, %arg1 : f32
    // CHECK: spirv.FOrdGreaterThanEqual
    %3 = spirv.FOrdGreaterThanEqual %arg0, %arg1 : f32
    // CHECK: spirv.FOrdLessThan
    %4 = spirv.FOrdLessThan %arg0, %arg1 : f32
    // CHECK: spirv.FOrdLessThanEqual
    %5 = spirv.FOrdLessThanEqual %arg0, %arg1 : f32
    // CHECK: spirv.FOrdNotEqual
    %6 = spirv.FOrdNotEqual %arg0, %arg1 : f32
    // CHECK: spirv.FUnordEqual
    %7 = spirv.FUnordEqual %arg0, %arg1 : f32
    // CHECK: spirv.FUnordGreaterThan
    %8 = spirv.FUnordGreaterThan %arg0, %arg1 : f32
    // CHECK: spirv.FUnordGreaterThanEqual
    %9 = spirv.FUnordGreaterThanEqual %arg0, %arg1 : f32
    // CHECK: spirv.FUnordLessThan
    %10 = spirv.FUnordLessThan %arg0, %arg1 : f32
    // CHECK: spirv.FUnordLessThanEqual
    %11 = spirv.FUnordLessThanEqual %arg0, %arg1 : f32
    // CHECK: spirv.FUnordNotEqual
    %12 = spirv.FUnordNotEqual %arg0, %arg1 : f32
    // CHECK: spirv.Ordered
    %13 = spirv.Ordered %arg0, %arg1 : f32
    // CHECK: spirv.Unordered
    %14 = spirv.Unordered %arg0, %arg1 : f32
    // CHECK: spirv.IsNan
    %15 = spirv.IsNan %arg0 : f32
    // CHECK: spirv.IsInf
    %16 = spirv.IsInf %arg1 : f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.SpecConstant @condition_scalar = true
  spirv.func @select() -> () "None" {
    %0 = spirv.Constant 4.0 : f32
    %1 = spirv.Constant 5.0 : f32
    %2 = spirv.mlir.referenceof @condition_scalar : i1
    // CHECK: spirv.Select {{.*}}, {{.*}}, {{.*}} : i1, f32
    %3 = spirv.Select %2, %0, %1 : i1, f32
    %4 = spirv.Constant dense<[2.0, 3.0, 4.0, 5.0]> : vector<4xf32>
    %5 = spirv.Constant dense<[6.0, 7.0, 8.0, 9.0]> : vector<4xf32>
    // CHECK: spirv.Select {{.*}}, {{.*}}, {{.*}} : i1, vector<4xf32>
    %6 = spirv.Select %2, %4, %5 : i1, vector<4xf32>
    %7 = spirv.Constant dense<[true, true, true, true]> : vector<4xi1>
    // CHECK: spirv.Select {{.*}}, {{.*}}, {{.*}} : vector<4xi1>, vector<4xf32>
    %8 = spirv.Select %7, %4, %5 : vector<4xi1>, vector<4xf32>
    spirv.Return
  }
}
