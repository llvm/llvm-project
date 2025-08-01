; RUN: llc -mtriple=amdgcn -mattr=-promote-alloca < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-promote-alloca < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}load_i8_sext_private:
; SI: buffer_load_sbyte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{$}}
define amdgpu_kernel void @load_i8_sext_private(ptr addrspace(1) %out) {
entry:
  %tmp0 = alloca i8, addrspace(5)
  %tmp1 = load i8, ptr addrspace(5) %tmp0
  %tmp2 = sext i8 %tmp1 to i32
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}load_i8_zext_private:
; SI: buffer_load_ubyte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{$}}
define amdgpu_kernel void @load_i8_zext_private(ptr addrspace(1) %out) {
entry:
  %tmp0 = alloca i8, addrspace(5)
  %tmp1 = load i8, ptr addrspace(5) %tmp0
  %tmp2 = zext i8 %tmp1 to i32
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_sext_private:
; SI: buffer_load_sshort v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{$}}
define amdgpu_kernel void @load_i16_sext_private(ptr addrspace(1) %out) {
entry:
  %tmp0 = alloca i16, addrspace(5)
  %tmp1 = load i16, ptr addrspace(5) %tmp0
  %tmp2 = sext i16 %tmp1 to i32
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_zext_private:
; SI: buffer_load_ushort v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 glc{{$}}
define amdgpu_kernel void @load_i16_zext_private(ptr addrspace(1) %out) {
entry:
  %tmp0 = alloca i16, addrspace(5)
  %tmp1 = load volatile i16, ptr addrspace(5) %tmp0
  %tmp2 = zext i16 %tmp1 to i32
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}
