; RUN: opt -mtriple=amdgcn -passes=amdgpu-lower-module-vgprs -S < %s | FileCheck %s

; A "VGPR as memory" global referenced only from an ordinary (non-kernel)
; function - as IPO might leave after outlining code from a kernel - is still
; laid out, and the referencing function is annotated. The backend handles
; direct references to the global from any function, independently of the
; frontend's placement rules.

; CHECK: @g = internal addrspace(13) global i32 poison, !amdgpu.vgpr.memory.offset
@g = internal addrspace(13) global i32 poison

; CHECK: define void @user(i32 %v) #[[ATTR:[0-9]+]]
define void @user(i32 %v) {
  store i32 %v, ptr addrspace(13) @g
  ret void
}

; CHECK: attributes #[[ATTR]] = {{.*}}"amdgpu-vgpr-memory-base"="{{[0-9]+}}"{{.*}}"amdgpu-vgpr-memory-size"="4"
