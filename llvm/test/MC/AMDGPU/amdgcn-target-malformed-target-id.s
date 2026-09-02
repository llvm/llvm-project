// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 %s -filetype=null 2>&1 | FileCheck %s

// Test malformed target ID error handling in .amdgcn_target directive
// CHECK: [[#@LINE+1]]:16: error: malformed target id 'not-a-valid-target-id'
.amdgcn_target "not-a-valid-target-id"
