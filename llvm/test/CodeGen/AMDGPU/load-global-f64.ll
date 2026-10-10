; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s
; RUN: llc -mtriple=amdgpu7.00-amdhsa < %s | FileCheck --check-prefixes=GCN-HSA,FUNC %s
; RUN: llc -mtriple=amdgpu8.02 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s

; FUNC-LABEL: {{^}}global_load_f64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-NOHSA: buffer_store_dwordx2 [[VAL]]

; GCN-HSA: flat_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-HSA: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]
define amdgpu_kernel void @global_load_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds double, ptr addrspace(1) %in, i32 %tid
  %ld = load double, ptr addrspace(1) %in.tid
  store double %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2f64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v2f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <2 x double>, ptr addrspace(1) %in, i32 %tid
  %ld = load <2 x double>, ptr addrspace(1) %in.tid
  store <2 x double> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3f64:
; GCN-NOHSA-DAG: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA-DAG: {{buffer|flat}}_load_dwordx2
; GCN-HSA-DAG: flat_load_dwordx4
; GCN-HSA-DAG: flat_load_dwordx2
define amdgpu_kernel void @global_load_v3f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <3 x double>, ptr addrspace(1) %in, i32 %tid
  %ld = load <3 x double>, ptr addrspace(1) %in.tid
  store <3 x double> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4f64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v4f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <4 x double>, ptr addrspace(1) %in, i32 %tid
  %ld = load <4 x double>, ptr addrspace(1) %in.tid
  store <4 x double> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8f64:
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4
; GCN-NOHSA: {{buffer|flat}}_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v8f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <8 x double>, ptr addrspace(1) %in, i32 %tid
  %ld = load <8 x double>, ptr addrspace(1) %in.tid
  store <8 x double> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16f64:
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
define amdgpu_kernel void @global_load_v16f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <16 x double>, ptr addrspace(1) %in, i32 %tid
  %ld = load <16 x double>, ptr addrspace(1) %in.tid
  store <16 x double> %ld, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
