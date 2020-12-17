// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @fmul(%arg0 : f32, %arg1 : f32) "None" {
    // CHECK: {{%.*}}= spv.FMul {{%.*}}, {{%.*}} : f32
    %0 = spv.FMul %arg0, %arg1 : f32
    spv.Return
  }
  spv.func @fadd(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  spv.func @fdiv(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FDiv {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FDiv %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  spv.func @fmod(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FMod {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FMod %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  spv.func @fnegate(%arg0 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FNegate {{%.*}} : vector<4xf32>
    %0 = spv.FNegate %arg0 : vector<4xf32>
    spv.Return
  }
  spv.func @fsub(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FSub {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FSub %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  spv.func @frem(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: {{%.*}} = spv.FRem {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FRem %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  spv.func @iadd(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.IAdd {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.IAdd %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @isub(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.ISub {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.ISub %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @imul(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.IMul {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.IMul %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @udiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.UDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.UDiv %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @umod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.UMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.UMod %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @sdiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.SDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SDiv %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @smod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.SMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SMod %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  spv.func @snegate(%arg0 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.SNegate {{%.*}} : vector<4xi32>
    %0 = spv.SNegate %arg0 : vector<4xi32>
    spv.Return
  }
  spv.func @srem(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) "None" {
    // CHECK: {{%.*}} = spv.SRem {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SRem %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
}
