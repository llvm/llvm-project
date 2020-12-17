// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
// CHECK-NEXT:   spv.func @foo() "Inline" {
// CHECK-NEXT:     spv.Return
// CHECK-NEXT:   }
// CHECK-NEXT: }

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @foo() -> () "Inline" {
     spv.Return
  }
}

// -----

// CHECK: v1.5
spv.module Logical GLSL450 requires #spv.vce<v1.5, [Shader], []> {
}

// -----

// CHECK: [Shader, Float16]
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader, Float16], []> {
}

// -----

// CHECK: [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]> {
}

