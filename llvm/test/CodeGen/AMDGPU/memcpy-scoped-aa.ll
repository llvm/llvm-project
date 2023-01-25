; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -stop-after=finalize-isel -o - %s | FileCheck --check-prefix=MIR %s

; Ensure that the scoped AA is attached on loads/stores lowered from mem ops.

; Re-evaluate the slot numbers of scopes as that numbering could be changed run-by-run.

; MIR-DAG: ![[DOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"bax"}
; MIR-DAG: ![[SCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %p"}
; MIR-DAG: ![[SCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %q"}
; MIR-DAG: ![[SET0:[0-9]+]] = !{![[SCOPE0]]}
; MIR-DAG: ![[SET1:[0-9]+]] = !{![[SCOPE1]]}

; MIR-LABEL: name: test_memcpy
; MIR: [[LOAD:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 %{{[0-9]+}}, 16, 0, implicit $exec :: (load (s128) from %ir.add.ptr, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
; MIR: GLOBAL_STORE_DWORDX4 %{{[0-9]+}}, killed [[LOAD]], 0, 0, implicit $exec :: (store (s128) into %ir.p, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
define i32 @test_memcpy(ptr addrspace(1) nocapture %p, ptr addrspace(1) nocapture readonly %q) {
; Check loads of %q are scheduled ahead of that store of the memcpy on %p.
; CHECK-LABEL: test_memcpy:
; CHECK-DAG:    global_load_dwordx2 v[[[Q0:[0-9]+]]:[[Q1:[0-9]+]]], v[2:3], off
; CHECK-DAG:    global_load_dwordx4 [[PVAL:v\[[0-9]+:[0-9]+\]]], v[0:1], off offset:16
; CHECK-DAG:    v_add_nc_u32_e32 v{{[0-9]+}}, v[[Q0]], v[[Q1]]
; CHECK:        global_store_dwordx4 v[0:1], [[PVAL]], off
; CHECK:        s_setpc_b64 s[30:31]
  %add.ptr = getelementptr inbounds i32, ptr addrspace(1) %p, i64 4
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %p, ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %add.ptr, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr addrspace(1) %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %v1 = load i32, ptr addrspace(1) %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memcpy_inline
; MIR: [[LOAD:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 %{{[0-9]+}}, 16, 0, implicit $exec :: (load (s128) from %ir.add.ptr, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
; MIR: GLOBAL_STORE_DWORDX4 %{{[0-9]+}}, killed [[LOAD]], 0, 0, implicit $exec :: (store (s128) into %ir.p, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
define i32 @test_memcpy_inline(ptr addrspace(1) nocapture %p, ptr addrspace(1) nocapture readonly %q) {
; Check loads of %q are scheduled ahead of that store of the memcpy on %p.
; CHECK-LABEL: test_memcpy_inline:
; CHECK-DAG:    global_load_dwordx2 v[[[Q0:[0-9]+]]:[[Q1:[0-9]+]]], v[2:3], off
; CHECK-DAG:    global_load_dwordx4 [[PVAL:v\[[0-9]+:[0-9]+\]]], v[0:1], off offset:16
; CHECK-DAG:    v_add_nc_u32_e32 v{{[0-9]+}}, v[[Q0]], v[[Q1]]
; CHECK:        global_store_dwordx4 v[0:1], [[PVAL]], off
; CHECK:        s_setpc_b64 s[30:31]
  %add.ptr = getelementptr inbounds i32, ptr addrspace(1) %p, i64 4
  tail call void @llvm.memcpy.inline.p1.p1.i64(ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %p, ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %add.ptr, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr addrspace(1) %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %v1 = load i32, ptr addrspace(1) %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memmove
; MIR: [[LOAD:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 %{{[0-9]+}}, 16, 0, implicit $exec :: (load (s128) from %ir.add.ptr, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
; MIR: GLOBAL_STORE_DWORDX4 %{{[0-9]+}}, killed [[LOAD]], 0, 0, implicit $exec :: (store (s128) into %ir.p, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
define i32 @test_memmove(ptr addrspace(1) nocapture %p, ptr addrspace(1) nocapture readonly %q) {
; Check loads of %q are scheduled ahead of that store of the memmove on %p.
; CHECK-LABEL: test_memmove:
; CHECK-DAG:    global_load_dwordx2 v[[[Q0:[0-9]+]]:[[Q1:[0-9]+]]], v[2:3], off
; CHECK-DAG:    global_load_dwordx4 [[PVAL:v\[[0-9]+:[0-9]+\]]], v[0:1], off offset:16
; CHECK-DAG:    v_add_nc_u32_e32 v{{[0-9]+}}, v[[Q0]], v[[Q1]]
; CHECK:        global_store_dwordx4 v[0:1], [[PVAL]]
; CHECK:        s_setpc_b64 s[30:31]
  %add.ptr = getelementptr inbounds i32, ptr addrspace(1) %p, i64 4
  tail call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %p, ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %add.ptr, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr addrspace(1) %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %v1 = load i32, ptr addrspace(1) %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memset
; MIR: GLOBAL_STORE_DWORDX4 killed %{{[0-9]+}}, killed %{{[0-9]+}}, 0, 0, implicit $exec :: (store (s128) into %ir.p, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]], addrspace 1)
define i32 @test_memset(ptr addrspace(1) nocapture %p, ptr addrspace(1) nocapture readonly %q) {
; Check loads of %q are scheduled ahead of that store of the memset on %p.
; CHECK-LABEL: test_memset:
; CHECK-DAG:    global_load_dwordx2 v[[[Q0:[0-9]+]]:[[Q1:[0-9]+]]], v[2:3], off
; CHECK-DAG:    v_mov_b32_e32 v[[PVAL:[0-9]+]], 0xaaaaaaaa
; CHECK:        global_store_dwordx4 v[0:1], v[[[PVAL]]{{:[0-9]+\]}}, off
; CHECK:        v_add_nc_u32_e32 v{{[0-9]+}}, v[[Q0]], v[[Q1]]
; CHECK:        s_setpc_b64 s[30:31]
  tail call void @llvm.memset.p1.i64(ptr addrspace(1) noundef nonnull align 4 dereferenceable(16) %p, i8 170, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr addrspace(1) %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %v1 = load i32, ptr addrspace(1) %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.inline.p1.p1.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p1.i64(ptr addrspace(1) nocapture writeonly, i8, i64, i1 immarg)

!0 = distinct !{!0, !"bax"}
!1 = distinct !{!1, !0, !"bax: %p"}
!2 = !{!1}
!3 = distinct !{!3, !0, !"bax: %q"}
!4 = !{!3}
