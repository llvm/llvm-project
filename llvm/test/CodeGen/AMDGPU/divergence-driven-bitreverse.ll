; RUN: llc -mtriple=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name: uniform_bitreverse_i32
; GCN: S_BREV_B32
define amdgpu_kernel void @uniform_bitreverse_i32(i32 %val, ptr addrspace(1) %out) {
  %res = call i32 @llvm.bitreverse.i32(i32 %val)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name: divergent_bitreverse_i32
; GCN: V_BFREV_B32
define amdgpu_kernel void @divergent_bitreverse_i32(i32 %val, ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %divergent = add i32 %val, %tid
  %res = call i32 @llvm.bitreverse.i32(i32 %divergent)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name: uniform_bitreverse_i64
; GCN: S_BREV_B64
define amdgpu_kernel void @uniform_bitreverse_i64(i64 %val, ptr addrspace(1) %out) {
  %res = call i64 @llvm.bitreverse.i64(i64 %val)
  store i64 %res, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name: divergent_bitreverse_i64
; GCN: V_BFREV_B32
; GCN: V_BFREV_B32
define amdgpu_kernel void @divergent_bitreverse_i64(i64 %val, ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ext = zext i32 %tid to i64
  %divergent = add i64 %val, %ext
  %res = call i64 @llvm.bitreverse.i64(i64 %divergent)
  store i64 %res, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
