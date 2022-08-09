// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.ControlBarrier
//===----------------------------------------------------------------------===//

func.func @control_barrier_0() -> () {
  // CHECK: spv.ControlBarrier <Workgroup>, <Device>, <Acquire|UniformMemory>
  spv.ControlBarrier <Workgroup>, <Device>, <Acquire|UniformMemory>
  return
}

// -----

func.func @control_barrier_1() -> () {
  // expected-error @+2 {{to be one of}}
  // expected-error @+1 {{failed to parse SPV_ScopeAttr}}
  spv.ControlBarrier <Something>, <Device>, <Acquire|UniformMemory>
  return
}


// -----

//===----------------------------------------------------------------------===//
// spv.MemoryBarrier
//===----------------------------------------------------------------------===//

func.func @memory_barrier_0() -> () {
  // CHECK: spv.MemoryBarrier <Device>, <Acquire|UniformMemory>
  spv.MemoryBarrier <Device>, <Acquire|UniformMemory>
  return
}

// -----

func.func @memory_barrier_1() -> () {
  // CHECK: spv.MemoryBarrier <Workgroup>, <Acquire>
  spv.MemoryBarrier <Workgroup>, <Acquire>
  return
}

// -----

func.func @memory_barrier_2() -> () {
 // expected-error @+1 {{expected at most one of these four memory constraints to be set: `Acquire`, `Release`,`AcquireRelease` or `SequentiallyConsistent`}}
  spv.MemoryBarrier <Device>, <Acquire|Release>
  return
}

