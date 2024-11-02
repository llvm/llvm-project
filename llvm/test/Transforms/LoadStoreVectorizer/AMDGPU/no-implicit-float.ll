; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; CHECK-LABEL: @no_implicit_float(
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
define amdgpu_kernel void @no_implicit_float(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 333, i32 addrspace(1)* %out.gep.3
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind noimplicitfloat }
