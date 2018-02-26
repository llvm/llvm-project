; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s


; GCN-LABEL: {{^}}set_inactive:
; GCN: s_not_b64 exec, exec
; GCN: v_mov_b32_e32 {{v[0-9]+}}, 42
; GCN: s_not_b64 exec, exec
define amdgpu_kernel void @set_inactive(i32 addrspace(1)* %out, i32 %in) {
  %tmp = call i32 @llvm.amdgcn.set.inactive.i32(i32 %in, i32 42) #0
  store i32 %tmp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}set_inactive_64:
; GCN: s_not_b64 exec, exec
; GCN: v_mov_b32_e32 {{v[0-9]+}}, 0
; GCN: v_mov_b32_e32 {{v[0-9]+}}, 0
; GCN: s_not_b64 exec, exec
define amdgpu_kernel void @set_inactive_64(i64 addrspace(1)* %out, i64 %in) {
  %tmp = call i64 @llvm.amdgcn.set.inactive.i64(i64 %in, i64 0) #0
  store i64 %tmp, i64 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) #0
declare i64 @llvm.amdgcn.set.inactive.i64(i64, i64) #0

attributes #0 = { convergent readnone }
