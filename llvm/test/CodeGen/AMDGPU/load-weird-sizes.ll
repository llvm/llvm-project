; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-NOHSA,SI,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=FUNC,CI-HSA,SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-NOHSA,SI,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=r600 -mcpu=cayman < %s | FileCheck -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}load_i24:
; SI-DAG: {{flat|buffer}}_load_ubyte
; SI-DAG: {{flat|buffer}}_load_ushort
; SI: {{flat|buffer}}_store_dword
define amdgpu_kernel void @load_i24(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %1 = load i24, ptr addrspace(1) %in
  %2 = zext i24 %1 to i32
  store i32 %2, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}load_i25:
; SI-NOHSA: buffer_load_dword [[VAL:v[0-9]+]]
; SI-NOHSA: buffer_store_dword [[VAL]]

; CI-HSA: flat_load_dword [[VAL:v[0-9]+]]
; CI-HSA: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VAL]]
define amdgpu_kernel void @load_i25(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %1 = load i25, ptr addrspace(1) %in
  %2 = zext i25 %1 to i32
  store i32 %2, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
