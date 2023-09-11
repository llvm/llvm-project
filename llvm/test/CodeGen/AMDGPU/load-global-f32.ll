; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC,SI-NOHSA %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-HSA,FUNC,GCNX3-HSA %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN-NOHSA,FUNC,GCNX3-NOHSA %s

; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=R600,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=cayman < %s | FileCheck --check-prefixes=R600,FUNC %s

; FUNC-LABEL: {{^}}global_load_f32:
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}
; GCN-HSA: flat_load_dword

; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0
define amdgpu_kernel void @global_load_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load float, ptr addrspace(1) %in
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2f32:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2

; R600: VTX_READ_64
define amdgpu_kernel void @global_load_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <2 x float>, ptr addrspace(1) %in
  store <2 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3f32:
; SI-NOHSA: buffer_load_dwordx4
; GCNX3-NOHSA: buffer_load_dwordx3
; GCNX3-HSA: flat_load_dwordx3

; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v3f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <3 x float>, ptr addrspace(1) %in
  store <3 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <4 x float>, ptr addrspace(1) %in
  store <4 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v8f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <8 x float>, ptr addrspace(1) %in
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v9f32:
; GCN-NOHSA: buffer_load_dword
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dword
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
; R600: VTX_READ_32
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v9f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <9 x float>, <9 x float> addrspace(1)* %in
  store <9 x float> %tmp0, <9 x float> addrspace(1)* %out
  ret void
}


; FUNC-LABEL: {{^}}global_load_v10f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx2

; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v10f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <10 x float>, <10 x float> addrspace(1)* %in
  store <10 x float> %tmp0, <10 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v11f32:
; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; GCNX3-NOHSA: buffer_load_dwordx4
; GCNX3-NOHSA: buffer_load_dwordx4
; GCNX3-NOHSA: buffer_load_dwordx3
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx3

; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v11f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <11 x float>, <11 x float> addrspace(1)* %in
  store <11 x float> %tmp0, <11 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v12f32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
define amdgpu_kernel void @global_load_v12f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <12 x float>, <12 x float> addrspace(1)* %in
  store <12 x float> %tmp0, <12 x float> addrspace(1)* %out
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
define amdgpu_kernel void @global_load_v16f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tmp0 = load <16 x float>, ptr addrspace(1) %in
  store <16 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
