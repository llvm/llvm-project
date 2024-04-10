; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -verify-machineinstrs -amdgpu-s-branch-bits=6 -amdgpu-long-branch-factor=0 < %s | FileCheck -check-prefix=GCN %s


; Restrict maximum branch to between +31 and -32 dwords
declare void @llvm.amdgcn.s.sleep(i32) #0

@name1 = external addrspace(1) global i32
@name2 = external addrspace(1) global i32
@name3 = external addrspace(1) global i32

; GCN-LABEL: {{^}}branch_offset_test:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0
; GCN-NEXT: s_cbranch_scc0 [[BB2:.LBB[0-9]+_[0-9]+]]
; GCN-NEXT: ; %bb.3: ; %bb
; GCN-NEXT: s_getpc_b64 s[[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]]
; GCN-NEXT: [[POST_GETPC:.Lpost_getpc[0-9]+]]:{{$}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], ([[BB3:.LBB[0-9]+_[0-9]+]]-[[POST_GETPC]])&4294967295
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], ([[BB3]]-[[POST_GETPC]])>>32
; GCN-NEXT: s_setpc_b64 s[[[PC_LO]]:[[PC_HI]]]
; GCN-NEXT: [[BB2]]: ; %bb2
; GCN-NEXT: s_getpc_b64 s[[[PC_LO]]:[[PC_HI]]]

; GCN: [[BB3]]: ; %bb3
define amdgpu_kernel void @branch_offset_test(ptr addrspace(1) %arg, i32 %cnd) #0 {
bb:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2:
  store i32 1, ptr addrspace(1) @name1
  store i32 2, ptr addrspace(1) @name2
  store i32 3, ptr addrspace(1) @name3
  call void @llvm.amdgcn.s.sleep(i32 0)
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}
