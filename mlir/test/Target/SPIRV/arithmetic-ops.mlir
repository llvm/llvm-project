// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @fmul(%arg0 : f32, %arg1 : f32) "None" {
    // CHECK: {{%.*}}= spirv.FMul {{%.*}}, {{%.*}} : f32
    %0 = spirv.FMul %arg0, %arg1 : f32
    spirv.Return
  }
  spirv.func @fadd(%arg0 : vector<5xf32>, %arg1 : vector<5xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : vector<5xf32>
    %0 = spirv.FAdd %arg0, %arg1 : vector<5xf32>
    spirv.Return
  }
  spirv.func @fdiv(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spirv.FDiv %arg0, %arg1 : vector<4xf32>
    spirv.Return
  }
  spirv.func @fmod(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FMod {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spirv.FMod %arg0, %arg1 : vector<4xf32>
    spirv.Return
  }
  spirv.func @fnegate(%arg0 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FNegate {{%.*}} : vector<4xf32>
    %0 = spirv.FNegate %arg0 : vector<4xf32>
    spirv.Return
  }
  spirv.func @fsub(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FSub {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spirv.FSub %arg0, %arg1 : vector<4xf32>
    spirv.Return
  }
  spirv.func @frem(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spirv.FRem {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spirv.FRem %arg0, %arg1 : vector<4xf32>
    spirv.Return
  }
  spirv.func @iadd(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.IAdd {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.IAdd %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @isub(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.ISub {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.ISub %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @imul(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.IMul {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.IMul %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @udiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.UDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.UDiv %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @umod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.UMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.UMod %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @sdiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.SDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.SDiv %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @smod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.SMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.SMod %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @snegate(%arg0 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.SNegate {{%.*}} : vector<4xi32>
    %0 = spirv.SNegate %arg0 : vector<4xi32>
    spirv.Return
  }
  spirv.func @srem(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spirv.SRem {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spirv.SRem %arg0, %arg1 : vector<4xi32>
    spirv.Return
  }
  spirv.func @vector_times_scalar(%arg0 : vector<4xf32>, %arg1 : f32) "None" {
    // CHECK: {{%.*}} = spirv.VectorTimesScalar {{%.*}}, {{%.*}} : (vector<4xf32>, f32) -> vector<4xf32>
    %0 = spirv.VectorTimesScalar %arg0, %arg1 : (vector<4xf32>, f32) -> vector<4xf32>
    spirv.Return
  }
}
