; RUN: llc -march=amdgcn -mattr=-promote-alloca -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}load_i8_sext_private:
; SI: buffer_load_sbyte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4{{$}}
define amdgpu_kernel void @load_i8_sext_private(i32 addrspace(1)* %out) {
entry:
  %tmp0 = alloca i8, addrspace(5)
  %tmp1 = load i8, i8 addrspace(5)* %tmp0
  %tmp2 = sext i8 %tmp1 to i32
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i8_zext_private:
; SI: buffer_load_ubyte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4{{$}}
define amdgpu_kernel void @load_i8_zext_private(i32 addrspace(1)* %out) {
entry:
  %tmp0 = alloca i8, addrspace(5)
  %tmp1 = load i8, i8 addrspace(5)* %tmp0
  %tmp2 = zext i8 %tmp1 to i32
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_sext_private:
; SI: buffer_load_sshort v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4{{$}}
define amdgpu_kernel void @load_i16_sext_private(i32 addrspace(1)* %out) {
entry:
  %tmp0 = alloca i16, addrspace(5)
  %tmp1 = load i16, i16 addrspace(5)* %tmp0
  %tmp2 = sext i16 %tmp1 to i32
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_zext_private:
; SI: buffer_load_ushort v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4 glc{{$}}
define amdgpu_kernel void @load_i16_zext_private(i32 addrspace(1)* %out) {
entry:
  %tmp0 = alloca i16, addrspace(5)
  %tmp1 = load volatile i16, i16 addrspace(5)* %tmp0
  %tmp2 = zext i16 %tmp1 to i32
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}
