; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 %s -o - | FileCheck %s

; CHECK: .amdhsa_code_object_version 6

define amdgpu_kernel void @kernel() {
  ret void
}
