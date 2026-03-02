; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @no_implicit_float(
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
define amdgpu_kernel void @no_implicit_float(ptr addrspace(1) %out) #0 {
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr i32, ptr addrspace(1) %out, i32 3

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out.gep.2
  store i32 333, ptr addrspace(1) %out.gep.3
  store i32 1234, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind noimplicitfloat }
