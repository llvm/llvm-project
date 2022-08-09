// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @memory_barrier_0() -> () "None" {
    // CHECK: spv.MemoryBarrier <Device>, <Release|UniformMemory>
    spv.MemoryBarrier <Device>, <Release|UniformMemory>
    spv.Return
  }
  spv.func @memory_barrier_1() -> () "None" {
    // CHECK: spv.MemoryBarrier <Subgroup>, <AcquireRelease|SubgroupMemory>
    spv.MemoryBarrier <Subgroup>, <AcquireRelease|SubgroupMemory>
    spv.Return
  }
  spv.func @control_barrier_0() -> () "None" {
    // CHECK: spv.ControlBarrier <Device>, <Workgroup>, <Release|UniformMemory>
    spv.ControlBarrier <Device>, <Workgroup>, <Release|UniformMemory>
    spv.Return
  }
  spv.func @control_barrier_1() -> () "None" {
    // CHECK: spv.ControlBarrier <Workgroup>, <Invocation>, <AcquireRelease|UniformMemory>
    spv.ControlBarrier <Workgroup>, <Invocation>, <AcquireRelease|UniformMemory>
    spv.Return
  }
}
