// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Geometry], []> {
  spirv.GlobalVariable @out : !spirv.ptr<!spirv.struct<(vector<4xf32>, f32, !spirv.array<1 x f32>)>, Output> 
  spirv.func @primitive_ops() "None" {
    // CHECK: spirv.EmitVertex
    spirv.EmitVertex
    // CHECK: spirv.EndPrimitive
    spirv.EndPrimitive
    spirv.Return
  }
  spirv.EntryPoint "Geometry" @primitive_ops, @out
  spirv.ExecutionMode @primitive_ops "InputPoints"
  spirv.ExecutionMode @primitive_ops "Invocations", 1
  spirv.ExecutionMode @primitive_ops "OutputLineStrip"
  spirv.ExecutionMode @primitive_ops "OutputVertices", 2
}
