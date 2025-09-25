// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel], []> {
  spirv.func @foo() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @foo
  // CHECK: spirv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  spirv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
}
