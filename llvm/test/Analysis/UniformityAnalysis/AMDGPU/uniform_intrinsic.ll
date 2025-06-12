; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: ALL VALUES UNIFORM
define amdgpu_kernel void @permlane64_constant(ptr addrspace(1) %out) {
  %v = call i32 @llvm.amdgcn.permlane64(i32 7)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: ALL VALUES UNIFORM
define amdgpu_kernel void @permlane64_uniform(ptr addrspace(1) %out, i32 %src) {
  %v = call i32 @llvm.amdgcn.permlane64(i32 %src)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlane64.i32(i32 %tid)
define amdgpu_kernel void @permlane64_nonuniform(i32 addrspace(1)* %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane64(i32 %tid)
  %out_ptr = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  store i32 %v, i32 addrspace(1)* %out_ptr
  ret void
}
