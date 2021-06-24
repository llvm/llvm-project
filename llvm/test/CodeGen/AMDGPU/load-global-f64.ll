; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-HSA,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC %s

; FUNC-LABEL: {{^}}global_load_f64:
; GCN-NOHSA: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-NOHSA: buffer_store_dwordx2 [[VAL]]

; GCN-HSA: flat_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-HSA: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]
define amdgpu_kernel void @global_load_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %ld = load double, double addrspace(1)* %in
  store double %ld, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2f64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in) #0 {
entry:
  %ld = load <2 x double>, <2 x double> addrspace(1)* %in
  store <2 x double> %ld, <2 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3f64:
; GCN-NOHSA-DAG: buffer_load_dwordx4
; GCN-NOHSA-DAG: buffer_load_dwordx2
; GCN-HSA-DAG: flat_load_dwordx4
; GCN-HSA-DAG: flat_load_dwordx2
define amdgpu_kernel void @global_load_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %in) #0 {
entry:
  %ld = load <3 x double>, <3 x double> addrspace(1)* %in
  store <3 x double> %ld, <3 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4f64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in) #0 {
entry:
  %ld = load <4 x double>, <4 x double> addrspace(1)* %in
  store <4 x double> %ld, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8f64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v8f64(<8 x double> addrspace(1)* %out, <8 x double> addrspace(1)* %in) #0 {
entry:
  %ld = load <8 x double>, <8 x double> addrspace(1)* %in
  store <8 x double> %ld, <8 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16f64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @global_load_v16f64(<16 x double> addrspace(1)* %out, <16 x double> addrspace(1)* %in) #0 {
entry:
  %ld = load <16 x double>, <16 x double> addrspace(1)* %in
  store <16 x double> %ld, <16 x double> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
