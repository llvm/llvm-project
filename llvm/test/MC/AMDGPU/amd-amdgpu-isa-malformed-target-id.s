// RUN: not llvm-mc -triple=amdgcn-amd-unknown -mcpu=gfx900 %s 2>&1 | FileCheck %s

// Test malformed target ID error handling in .amd_amdgpu_isa directive
// CHECK: [[#@LINE+1]]:17: error: malformed target id 'not-a-valid-target-id'
.amd_amdgpu_isa "not-a-valid-target-id"
