// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-NEXT:   spirv.func @foo() "Inline" {
// CHECK-NEXT:     spirv.Return
// CHECK-NEXT:   }
// CHECK-NEXT: }

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "Inline" {
     spirv.Return
  }
}

// -----

// CHECK: v1.5
spirv.module Logical GLSL450 requires #spirv.vce<v1.5, [Shader], []> {
}

// -----

// CHECK: [Shader, Float16]
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Float16], []> {
}

// -----

// CHECK: [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]> {
}

