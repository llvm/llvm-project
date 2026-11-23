// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, DemoteToHelperInvocation], [SPV_EXT_demote_to_helper_invocation]> {
  // CHECK-LABEL: @demote
  spirv.func @demote() -> () "None" {
    // CHECK: spirv.DemoteToHelperInvocation
    spirv.DemoteToHelperInvocation
    spirv.Return
  }
  spirv.EntryPoint "Fragment" @demote
  spirv.ExecutionMode @demote "OriginUpperLeft"
}
