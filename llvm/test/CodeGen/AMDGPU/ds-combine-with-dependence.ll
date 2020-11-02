; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s


; There is no dependence between the store and the two loads. So we can combine
; the loads and schedule it freely.

; GCN-LABEL: {{^}}ds_combine_nodep

; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-DAG: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:7 offset1:8
; GCN: s_waitcnt lgkmcnt({{[0-9]+}})
define amdgpu_kernel void @ds_combine_nodep(float addrspace(1)* %out, float addrspace(3)* %inptr) {

  %base = bitcast float addrspace(3)* %inptr to i8 addrspace(3)*
  %addr0 = getelementptr i8, i8 addrspace(3)* %base, i32 24
  %tmp0 = bitcast i8 addrspace(3)* %addr0 to float addrspace(3)*
  %vaddr0 = bitcast float addrspace(3)* %tmp0 to <3 x float> addrspace(3)*
  %load0 = load <3 x float>, <3 x float> addrspace(3)* %vaddr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %vaddrs = bitcast float addrspace(3)* %tmp2 to <2 x float> addrspace(3)*
  store <2 x float> %data, <2 x float> addrspace(3)* %vaddrs, align 4

  %vaddr1 = getelementptr float, float addrspace(3)* %inptr, i32 7
  %v1 = load float, float addrspace(3)* %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, float addrspace(1)* %out, align 4
  ret void
}


; The store depends on the first load, so we could not move the first load down to combine with
; the second load directly. However, we can move the store after the combined load.

; GCN-LABEL: {{^}}ds_combine_WAR

; GCN:      ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:7 offset1:27
; GCN-NEXT: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
define amdgpu_kernel void @ds_combine_WAR(float addrspace(1)* %out, float addrspace(3)* %inptr) {

  %base = bitcast float addrspace(3)* %inptr to i8 addrspace(3)*
  %addr0 = getelementptr i8, i8 addrspace(3)* %base, i32 100
  %tmp0 = bitcast i8 addrspace(3)* %addr0 to float addrspace(3)*
  %vaddr0 = bitcast float addrspace(3)* %tmp0 to <3 x float> addrspace(3)*
  %load0 = load <3 x float>, <3 x float> addrspace(3)* %vaddr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %vaddrs = bitcast float addrspace(3)* %tmp2 to <2 x float> addrspace(3)*
  store <2 x float> %data, <2 x float> addrspace(3)* %vaddrs, align 4

  %vaddr1 = getelementptr float, float addrspace(3)* %inptr, i32 7
  %v1 = load float, float addrspace(3)* %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, float addrspace(1)* %out, align 4
  ret void
}


; The second load depends on the store. We can combine the two loads, and the combined load is
; at the original place of the second load.

; GCN-LABEL: {{^}}ds_combine_RAW

; GCN:      ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-NEXT: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:8 offset1:26
define amdgpu_kernel void @ds_combine_RAW(float addrspace(1)* %out, float addrspace(3)* %inptr) {

  %base = bitcast float addrspace(3)* %inptr to i8 addrspace(3)*
  %addr0 = getelementptr i8, i8 addrspace(3)* %base, i32 24
  %tmp0 = bitcast i8 addrspace(3)* %addr0 to float addrspace(3)*
  %vaddr0 = bitcast float addrspace(3)* %tmp0 to <3 x float> addrspace(3)*
  %load0 = load <3 x float>, <3 x float> addrspace(3)* %vaddr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %vaddrs = bitcast float addrspace(3)* %tmp2 to <2 x float> addrspace(3)*
  store <2 x float> %data, <2 x float> addrspace(3)* %vaddrs, align 4

  %vaddr1 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %v1 = load float, float addrspace(3)* %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, float addrspace(1)* %out, align 4
  ret void
}


; The store depends on the first load, also the second load depends on the store.
; So we can not combine the two loads.

; GCN-LABEL: {{^}}ds_combine_WAR_RAW

; GCN:      ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:108
; GCN-NEXT: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-NEXT: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:104
define amdgpu_kernel void @ds_combine_WAR_RAW(float addrspace(1)* %out, float addrspace(3)* %inptr) {

  %base = bitcast float addrspace(3)* %inptr to i8 addrspace(3)*
  %addr0 = getelementptr i8, i8 addrspace(3)* %base, i32 100
  %tmp0 = bitcast i8 addrspace(3)* %addr0 to float addrspace(3)*
  %vaddr0 = bitcast float addrspace(3)* %tmp0 to <3 x float> addrspace(3)*
  %load0 = load <3 x float>, <3 x float> addrspace(3)* %vaddr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %vaddrs = bitcast float addrspace(3)* %tmp2 to <2 x float> addrspace(3)*
  store <2 x float> %data, <2 x float> addrspace(3)* %vaddrs, align 4

  %vaddr1 = getelementptr float, float addrspace(3)* %inptr, i32 26
  %v1 = load float, float addrspace(3)* %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, float addrspace(1)* %out, align 4
  ret void
}
