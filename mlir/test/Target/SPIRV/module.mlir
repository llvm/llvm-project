// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip --split-input-file %s | FileCheck %s

// REQUIRES: shell
// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t && mlir-translate --no-implicit-module --serialize-spirv \ %}
// RUN: %if spirv-tools %{ --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s \ %}
// RUN: %if spirv-tools %{ && ls %t/module* | xargs -I{} bash -c 'spirv-val {}' %}

// CHECK:      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-NEXT:   spirv.func @foo() "Inline" {
// CHECK-NEXT:     spirv.Return
// CHECK-NEXT:   }
// CHECK-NEXT:   spirv.EntryPoint "Vertex" @foo
// CHECK-NEXT: }

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "Inline" {
     spirv.Return
  }
  spirv.EntryPoint "Vertex" @foo
}

// -----

// CHECK: v1.5
spirv.module Logical GLSL450 requires #spirv.vce<v1.5, [Shader, Linkage], []> {
}

// -----

// CHECK: v1.6
spirv.module Logical GLSL450 requires #spirv.vce<v1.6, [Shader, Linkage], []> {
}

// -----

// CHECK: [Shader, Float16, Linkage]
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Float16, Linkage], []> {
}

// -----

// CHECK: [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_float_controls, SPV_KHR_subgroup_vote]> {
}

