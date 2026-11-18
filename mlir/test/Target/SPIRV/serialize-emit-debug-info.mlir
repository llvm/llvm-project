// RUN: mlir-translate --no-implicit-module --serialize-spirv --spirv-emit-debug-info %s | mlir-translate --deserialize-spirv --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s --check-prefix=DEBUG
// RUN: mlir-translate --no-implicit-module --serialize-spirv %s | mlir-translate --deserialize-spirv --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s --check-prefix=NODEBUG
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --spirv-emit-debug-info %s | spirv-val %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_non_semantic_info]> {
  spirv.func @main() "None" {
    %cst = spirv.Constant 1.0 : f32
    // DEBUG: spirv.FAdd {{.*}} loc({{".*serialize-emit-debug-info.mlir"}}:10:10)
    // NODEBUG: spirv.FAdd {{.*}} loc(unknown)
    %0 = spirv.FAdd %cst, %cst : f32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}
