; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s
; RUN: llc -mtriple=amdgpu7.00--amdhsa < %s | FileCheck --check-prefixes=GCN-HSA,FUNC %s
; RUN: llc -mtriple=amdgpu8.02 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s

; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=EG,FUNC %s
; RUN: llc -mtriple=r600 -mcpu=cayman < %s | FileCheck --check-prefixes=EG,FUNC %s

; FUNC-LABEL: {{^}}global_load_i64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-NOHSA: buffer_store_dwordx2 [[VAL]]

; GCN-HSA: flat_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-HSA: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]

; EG: VTX_READ_64
define amdgpu_kernel void @global_load_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds i64, ptr addrspace(1) %in, i32 %tid
  %ld = load i64, ptr addrspace(1) %in.tid
  store i64 %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2i64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
define amdgpu_kernel void @global_load_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <2 x i64>, ptr addrspace(1) %in, i32 %tid
  %ld = load <2 x i64>, ptr addrspace(1) %in.tid
  store <2 x i64> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3i64:
; GCN-NOHSA-DAG: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA-DAG: {{buffer|flat}}_load_dwordx2

; GCN-HSA-DAG: flat_load_dwordx4
; GCN-HSA-DAG: flat_load_dwordx2

; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @global_load_v3i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <3 x i64>, ptr addrspace(1) %in, i32 %tid
  %ld = load <3 x i64>, ptr addrspace(1) %in.tid
  store <3 x i64> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4i64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @global_load_v4i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <4 x i64>, ptr addrspace(1) %in, i32 %tid
  %ld = load <4 x i64>, ptr addrspace(1) %in.tid
  store <4 x i64> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8i64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @global_load_v8i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <8 x i64>, ptr addrspace(1) %in, i32 %tid
  %ld = load <8 x i64>, ptr addrspace(1) %in.tid
  store <8 x i64> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16i64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @global_load_v16i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <16 x i64>, ptr addrspace(1) %in, i32 %tid
  %ld = load <16 x i64>, ptr addrspace(1) %in.tid
  store <16 x i64> %ld, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
