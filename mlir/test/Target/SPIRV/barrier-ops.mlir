// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
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
