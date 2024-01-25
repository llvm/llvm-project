; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s


; There is no dependence between the store and the two loads. So we can combine
; the loads and schedule it freely.

; GCN-LABEL: {{^}}ds_combine_nodep

; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-DAG: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:7 offset1:8
; GCN: s_waitcnt lgkmcnt({{[0-9]+}})
define amdgpu_kernel void @ds_combine_nodep(ptr addrspace(1) %out, ptr addrspace(3) %inptr) {

  %addr0 = getelementptr i8, ptr addrspace(3) %inptr, i32 24
  %load0 = load <3 x float>, ptr addrspace(3) %addr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  store <2 x float> %data, ptr addrspace(3) %tmp2, align 4

  %vaddr1 = getelementptr float, ptr addrspace(3) %inptr, i32 7
  %v1 = load float, ptr addrspace(3) %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}


; The store depends on the first load, so we could not move the first load down to combine with
; the second load directly. However, we can move the store after the combined load.

; GCN-LABEL: {{^}}ds_combine_WAR

; GCN:      ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:7 offset1:27
; GCN-NEXT: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
define amdgpu_kernel void @ds_combine_WAR(ptr addrspace(1) %out, ptr addrspace(3) %inptr) {

  %addr0 = getelementptr i8, ptr addrspace(3) %inptr, i32 100
  %load0 = load <3 x float>, ptr addrspace(3) %addr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  store <2 x float> %data, ptr addrspace(3) %tmp2, align 4

  %vaddr1 = getelementptr float, ptr addrspace(3) %inptr, i32 7
  %v1 = load float, ptr addrspace(3) %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}


; The second load depends on the store. We could combine the two loads, putting
; the combined load at the original place of the second load, but we prefer to
; leave the first load near the start of the function to hide its latency.

; GCN-LABEL: {{^}}ds_combine_RAW

; GCN:      ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-NEXT: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:32
; GCN-NEXT: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:104
define amdgpu_kernel void @ds_combine_RAW(ptr addrspace(1) %out, ptr addrspace(3) %inptr) {

  %addr0 = getelementptr i8, ptr addrspace(3) %inptr, i32 24
  %load0 = load <3 x float>, ptr addrspace(3) %addr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  store <2 x float> %data, ptr addrspace(3) %tmp2, align 4

  %vaddr1 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  %v1 = load float, ptr addrspace(3) %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}


; The store depends on the first load, also the second load depends on the store.
; So we can not combine the two loads.

; GCN-LABEL: {{^}}ds_combine_WAR_RAW

; GCN:      ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:108
; GCN-NEXT: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:26 offset1:27
; GCN-NEXT: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:104
define amdgpu_kernel void @ds_combine_WAR_RAW(ptr addrspace(1) %out, ptr addrspace(3) %inptr) {

  %addr0 = getelementptr i8, ptr addrspace(3) %inptr, i32 100
  %load0 = load <3 x float>, ptr addrspace(3) %addr0, align 4
  %v0 = extractelement <3 x float> %load0, i32 2

  %tmp1 = insertelement <2 x float> undef, float 1.0, i32 0
  %data = insertelement <2 x float> %tmp1, float 2.0, i32 1

  %tmp2 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  store <2 x float> %data, ptr addrspace(3) %tmp2, align 4

  %vaddr1 = getelementptr float, ptr addrspace(3) %inptr, i32 26
  %v1 = load float, ptr addrspace(3) %vaddr1, align 4

  %sum = fadd float %v0, %v1
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}
