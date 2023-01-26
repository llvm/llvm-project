; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.sffbh.i32(i32) #1

; GCN-LABEL: {{^}}s_flbit:
; GCN: s_load_dword [[VAL:s[0-9]+]],
; GCN: s_flbit_i32 [[SRESULT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_dword [[VRESULT]],
define amdgpu_kernel void @s_flbit(ptr addrspace(1) noalias %out, i32 %val) #0 {
  %r = call i32 @llvm.amdgcn.sffbh.i32(i32 %val)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_flbit:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_ffbh_i32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[RESULT]],
define amdgpu_kernel void @v_flbit(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %valptr) #0 {
  %val = load i32, ptr addrspace(1) %valptr, align 4
  %r = call i32 @llvm.amdgcn.sffbh.i32(i32 %val)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
