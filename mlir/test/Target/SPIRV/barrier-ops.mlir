// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.func @memory_barrier_0() -> () "None" {
    // CHECK: spirv.MemoryBarrier <Device>, <Release|UniformMemory>
    spirv.MemoryBarrier <Device>, <Release|UniformMemory>
    spirv.Return
  }
  spirv.func @memory_barrier_1() -> () "None" {
    // CHECK: spirv.MemoryBarrier <Subgroup>, <AcquireRelease|SubgroupMemory>
    spirv.MemoryBarrier <Subgroup>, <AcquireRelease|SubgroupMemory>
    spirv.Return
  }
  spirv.func @control_barrier_0() -> () "None" {
    // CHECK: spirv.ControlBarrier <Device>, <Workgroup>, <Release|UniformMemory>
    spirv.ControlBarrier <Device>, <Workgroup>, <Release|UniformMemory>
    spirv.Return
  }
  spirv.func @control_barrier_1() -> () "None" {
    // CHECK: spirv.ControlBarrier <Workgroup>, <Invocation>, <AcquireRelease|UniformMemory>
    spirv.ControlBarrier <Workgroup>, <Invocation>, <AcquireRelease|UniformMemory>
    spirv.Return
  }
}
