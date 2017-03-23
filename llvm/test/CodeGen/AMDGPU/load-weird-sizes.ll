; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NOHSA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=CI-HSA -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NOHSA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=R600 -check-prefix=CM -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}load_i24:
; SI: {{flat|buffer}}_load_ubyte
; SI: {{flat|buffer}}_load_ushort
; SI: {{flat|buffer}}_store_dword
define amdgpu_kernel void @load_i24(i32 addrspace(1)* %out, i24 addrspace(1)* %in) #0 {
  %1 = load i24, i24 addrspace(1)* %in
  %2 = zext i24 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i25:
; SI-NOHSA: buffer_load_dword [[VAL:v[0-9]+]]
; SI-NOHSA: buffer_store_dword [[VAL]]

; CI-HSA: flat_load_dword [[VAL:v[0-9]+]]
; CI-HSA: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VAL]]
define amdgpu_kernel void @load_i25(i32 addrspace(1)* %out, i25 addrspace(1)* %in) #0 {
  %1 = load i25, i25 addrspace(1)* %in
  %2 = zext i25 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
