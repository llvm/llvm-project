; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC,SI-NOHSA %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-HSA,FUNC,GCNX3-HSA %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC,GCNX3-NOHSA %s

; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=R600,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=cayman < %s | FileCheck --check-prefixes=R600,FUNC %s

; FUNC-LABEL: {{^}}global_load_f32:
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}
; GCN-HSA: flat_load_dword

; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0
define amdgpu_kernel void @global_load_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %tmp0 = load float, float addrspace(1)* %in
  store float %tmp0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2f32:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2

; R600: VTX_READ_64
define amdgpu_kernel void @global_load_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in) #0 {
entry:
  %tmp0 = load <2 x float>, <2 x float> addrspace(1)* %in
  store <2 x float> %tmp0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3f32:
; SI-NOHSA: buffer_load_dwordx4
; GCNX3-NOHSA: buffer_load_dwordx3
; GCNX3-HSA: flat_load_dwordx3

; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %in) #0 {
entry:
  %tmp0 = load <3 x float>, <3 x float> addrspace(1)* %in
  store <3 x float> %tmp0, <3 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) #0 {
entry:
  %tmp0 = load <4 x float>, <4 x float> addrspace(1)* %in
  store <4 x float> %tmp0, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v8f32(<8 x float> addrspace(1)* %out, <8 x float> addrspace(1)* %in) #0 {
entry:
  %tmp0 = load <8 x float>, <8 x float> addrspace(1)* %in
  store <8 x float> %tmp0, <8 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v16f32(<16 x float> addrspace(1)* %out, <16 x float> addrspace(1)* %in) #0 {
entry:
  %tmp0 = load <16 x float>, <16 x float> addrspace(1)* %in
  store <16 x float> %tmp0, <16 x float> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
