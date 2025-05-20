; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx90a -O0 -print-after=legalizer %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-LABEL: widen_ashr_i4:
define amdgpu_kernel void @widen_ashr_i4(
    ptr addrspace(1) %res, i4 %a, i4 %b) {
; CHECK: G_ASHR %{{[0-9]+}}:_, %{{[0-9]+}}:_(s16)
entry:
  %res.val = ashr i4 %a, %b
  store i4 %res.val, ptr addrspace(1) %res
  ret void
}

; CHECK-LABEL: widen_ashr_v4i1:
define amdgpu_kernel void @widen_ashr_v4i1(
    ptr addrspace(1) %res, <4 x i1> %a, <4 x i1> %b) {
; CHECK: G_ASHR %{{[0-9]+}}:_, %{{[0-9]+}}:_(s16)
; CHECK: G_ASHR %{{[0-9]+}}:_, %{{[0-9]+}}:_(s16)
; CHECK: G_ASHR %{{[0-9]+}}:_, %{{[0-9]+}}:_(s16)
; CHECK: G_ASHR %{{[0-9]+}}:_, %{{[0-9]+}}:_(s16)
entry:
  %res.val = ashr <4 x i1> %a, %b
  store <4 x i1> %res.val, ptr addrspace(1) %res
  ret void
}
