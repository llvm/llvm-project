; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}short_char:
; GCN: global_store_byte v
define protected amdgpu_kernel void @short_char(i8 addrspace(1)* %out) {
entry:
  %tmp = load i16, i16 addrspace(1)* undef
  %tmp1 = trunc i16 %tmp to i8
  store i8 %tmp1, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}short2_char4:
; GCN: global_store_dword v
define protected amdgpu_kernel void @short2_char4(<4 x i8> addrspace(1)* %out) {
entry:
  %tmp = load <2 x i16>, <2 x i16> addrspace(1)* undef, align 4
  %vecinit = shufflevector <2 x i16> %tmp, <2 x i16> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %vecinit2 = shufflevector <4 x i16> %vecinit, <4 x i16> <i16 undef, i16 undef, i16 0, i16 0>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %tmp1 = trunc <4 x i16> %vecinit2 to <4 x i8>
  store <4 x i8> %tmp1, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}short4_char8:
; GCN: global_store_dwordx2 v
define protected amdgpu_kernel void @short4_char8(<8 x i8> addrspace(1)* %out) {
entry:
  %tmp = load <4 x i16>, <4 x i16> addrspace(1)* undef, align 8
  %vecinit = shufflevector <4 x i16> %tmp, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2 = shufflevector <8 x i16> %vecinit, <8 x i16> <i16 undef, i16 undef, i16 undef, i16 undef, i16 0, i16 0, i16 0, i16 0>, <8 x i32> <i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7>
  %tmp1 = trunc <8 x i16> %vecinit2 to <8 x i8>
  store <8 x i8> %tmp1, <8 x i8> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}short8_char16:
; GCN: global_store_dwordx4 v
define protected amdgpu_kernel void @short8_char16(<16 x i8> addrspace(1)* %out) {
entry:
  %tmp = load <8 x i16>, <8 x i16> addrspace(1)* undef, align 16
  %vecinit = shufflevector <8 x i16> %tmp, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2 = shufflevector <16 x i16> %vecinit, <16 x i16> <i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <16 x i32> <i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7>
  %tmp1 = trunc <16 x i16> %vecinit2 to <16 x i8>
  store <16 x i8> %tmp1, <16 x i8> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}short16_char32:
; GCN: global_store_dwordx4 v
; GCN: global_store_dwordx4 v
define protected amdgpu_kernel void @short16_char32(<32 x i8> addrspace(1)* %out) {
entry:
  %tmp = load <16 x i16>, <16 x i16> addrspace(1)* undef, align 32
  %vecinit = shufflevector <16 x i16> %tmp, <16 x i16> undef, <32 x i32> <i32 0, i32 1, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2 = shufflevector <32 x i16> %vecinit, <32 x i16> <i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 0, i16 1, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 undef, i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <32 x i32> <i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7, i32 0, i32 1, i32 6, i32 7>
  %tmp1 = trunc <32 x i16> %vecinit2 to <32 x i8>
  store <32 x i8> %tmp1, <32 x i8> addrspace(1)* %out, align 32
  ret void
}
