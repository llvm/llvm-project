// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @iequal_scalar(%arg0: i32, %arg1: i32)  "None" {
    // CHECK: {{.*}} = spv.IEqual {{.*}}, {{.*}} : i32
    %0 = spv.IEqual %arg0, %arg1 : i32
    spv.Return
  }
  spv.func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.INotEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SGreaterThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SLessThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.UGreaterThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
    // CHECK: {{.*}} = spv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.ULessThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>)  "None" {
    // CHECK: {{.*}} = spv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @cmpf(%arg0 : f32, %arg1 : f32) "None" {
    // CHECK: spv.FOrdEqual
    %1 = spv.FOrdEqual %arg0, %arg1 : f32
    // CHECK: spv.FOrdGreaterThan
    %2 = spv.FOrdGreaterThan %arg0, %arg1 : f32
    // CHECK: spv.FOrdGreaterThanEqual
    %3 = spv.FOrdGreaterThanEqual %arg0, %arg1 : f32
    // CHECK: spv.FOrdLessThan
    %4 = spv.FOrdLessThan %arg0, %arg1 : f32
    // CHECK: spv.FOrdLessThanEqual
    %5 = spv.FOrdLessThanEqual %arg0, %arg1 : f32
    // CHECK: spv.FOrdNotEqual
    %6 = spv.FOrdNotEqual %arg0, %arg1 : f32
    // CHECK: spv.FUnordEqual
    %7 = spv.FUnordEqual %arg0, %arg1 : f32
    // CHECK: spv.FUnordGreaterThan
    %8 = spv.FUnordGreaterThan %arg0, %arg1 : f32
    // CHECK: spv.FUnordGreaterThanEqual
    %9 = spv.FUnordGreaterThanEqual %arg0, %arg1 : f32
    // CHECK: spv.FUnordLessThan
    %10 = spv.FUnordLessThan %arg0, %arg1 : f32
    // CHECK: spv.FUnordLessThanEqual
    %11 = spv.FUnordLessThanEqual %arg0, %arg1 : f32
    // CHECK: spv.FUnordNotEqual
    %12 = spv.FUnordNotEqual %arg0, %arg1 : f32
    // CHECK: spv.Ordered
    %13 = spv.Ordered %arg0, %arg1 : f32
    // CHECK: spv.Unordered
    %14 = spv.Unordered %arg0, %arg1 : f32
    // CHCK: spv.IsNan
    %15 = spv.IsNan %arg0 : f32
    // CHCK: spv.IsInf
    %16 = spv.IsInf %arg1 : f32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.SpecConstant @condition_scalar = true
  spv.func @select() -> () "None" {
    %0 = spv.constant 4.0 : f32
    %1 = spv.constant 5.0 : f32
    %2 = spv.mlir.referenceof @condition_scalar : i1
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : i1, f32
    %3 = spv.Select %2, %0, %1 : i1, f32
    %4 = spv.constant dense<[2.0, 3.0, 4.0, 5.0]> : vector<4xf32>
    %5 = spv.constant dense<[6.0, 7.0, 8.0, 9.0]> : vector<4xf32>
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : i1, vector<4xf32>
    %6 = spv.Select %2, %4, %5 : i1, vector<4xf32>
    %7 = spv.constant dense<[true, true, true, true]> : vector<4xi1>
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : vector<4xi1>, vector<4xf32>
    %8 = spv.Select %7, %4, %5 : vector<4xi1>, vector<4xf32>
    spv.Return
  }
}
