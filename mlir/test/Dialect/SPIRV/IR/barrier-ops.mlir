// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ControlBarrier
//===----------------------------------------------------------------------===//

func.func @control_barrier_0() -> () {
  // CHECK: spirv.ControlBarrier <Workgroup>, <Device>, <Acquire|UniformMemory>
  spirv.ControlBarrier <Workgroup>, <Device>, <Acquire|UniformMemory>
  return
}

// -----

func.func @control_barrier_1() -> () {
  // expected-error @+2 {{to be one of}}
  // expected-error @+1 {{failed to parse SPIRV_ScopeAttr}}
  spirv.ControlBarrier <Something>, <Device>, <Acquire|UniformMemory>
  return
}


// -----

//===----------------------------------------------------------------------===//
// spirv.MemoryBarrier
//===----------------------------------------------------------------------===//

func.func @memory_barrier_0() -> () {
  // CHECK: spirv.MemoryBarrier <Device>, <Acquire|UniformMemory>
  spirv.MemoryBarrier <Device>, <Acquire|UniformMemory>
  return
}

// -----

func.func @memory_barrier_1() -> () {
  // CHECK: spirv.MemoryBarrier <Workgroup>, <Acquire>
  spirv.MemoryBarrier <Workgroup>, <Acquire>
  return
}

// -----

func.func @memory_barrier_2() -> () {
 // expected-error @+1 {{expected at most one of these four memory constraints to be set: `Acquire`, `Release`,`AcquireRelease` or `SequentiallyConsistent`}}
  spirv.MemoryBarrier <Device>, <Acquire|Release>
  return
}

