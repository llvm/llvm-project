; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa %s -o - | FileCheck %s

; CHECK: .amdhsa_code_object_version 6

define amdgpu_kernel void @kernel() {
  ret void
}
