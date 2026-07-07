; RUN: llc -mtriple=amdgpu12.00 -mattr=+real-true16 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GFX12 %s
; RUN: llc -global-isel -mtriple=amdgpu12.00 -mattr=+real-true16 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GFX12 %s

; Check that only the _t16 suffixed forms of the loads are selected for both
; the SelectionDAG and GlobalISel paths.

define <8 x i16> @use_t16_global(i32 %off, ptr addrspace(1) %ptr) {
; GFX12-LABEL: name: use_t16_global
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: GLOBAL_LOAD_SHORT_D16_t16 %{{[0-9]+}}
  %ext = zext i32 %off to i64
  %gep = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %ptr, i64 %ext
  %ld0 = load i16, ptr addrspace(1) %gep, align 2, !tbaa !0
  %v0 = insertelement <8 x i16> poison, i16 %ld0, i64 0
  %gep1 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 512
  %ld1 = load i16, ptr addrspace(1) %gep1, align 2, !tbaa !0
  %v1 = insertelement <8 x i16> %v0, i16 %ld1, i64 1
  %gep2 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 1024
  %ld2 = load i16, ptr addrspace(1) %gep2, align 2, !tbaa !0
  %v2 = insertelement <8 x i16> %v1, i16 %ld2, i64 2
  %gep3 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 1536
  %ld3 = load i16, ptr addrspace(1) %gep3, align 2, !tbaa !0
  %v3 = insertelement <8 x i16> %v2, i16 %ld3, i64 3
  %gep4 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 2048
  %ld4 = load i16, ptr addrspace(1) %gep4, align 2, !tbaa !0
  %v4 = insertelement <8 x i16> %v3, i16 %ld4, i64 4
  %gep5 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 2560
  %ld5 = load i16, ptr addrspace(1) %gep5, align 2, !tbaa !0
  %v5 = insertelement <8 x i16> %v4, i16 %ld5, i64 5
  %gep6 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 3072
  %ld6 = load i16, ptr addrspace(1) %gep6, align 2, !tbaa !0
  %v6 = insertelement <8 x i16> %v5, i16 %ld6, i64 6
  %gep7 = getelementptr inbounds nuw i8, ptr addrspace(1) %gep, i64 3584
  %ld7 = load i16, ptr addrspace(1) %gep7, align 2, !tbaa !0
  %v7 = insertelement <8 x i16> %v6, i16 %ld7, i64 7
  %gep8 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %ptr, i64 10
  %ld8 = load i16, ptr addrspace(1) %gep8, align 2, !tbaa !0
  %res = insertelement <8 x i16> %v6, i16 %ld8, i64 1
  ret <8 x i16> %res
}

define <8 x i16> @use_t16_scratch(i32 %off, ptr addrspace(5) %ptr) {
; GFX12-LABEL: name: use_t16_scratch
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: SCRATCH_LOAD_SHORT_D16_t16 %{{[0-9]+}}
  %gep = getelementptr inbounds nuw [2 x i8], ptr addrspace(5) %ptr, i32 %off
  %ld0 = load i16, ptr addrspace(5) %gep, align 2, !tbaa !0
  %v0 = insertelement <8 x i16> poison, i16 %ld0, i64 0
  %gep1 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 512
  %ld1 = load i16, ptr addrspace(5) %gep1, align 2, !tbaa !0
  %v1 = insertelement <8 x i16> %v0, i16 %ld1, i64 1
  %gep2 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 1024
  %ld2 = load i16, ptr addrspace(5) %gep2, align 2, !tbaa !0
  %v2 = insertelement <8 x i16> %v1, i16 %ld2, i64 2
  %gep3 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 1536
  %ld3 = load i16, ptr addrspace(5) %gep3, align 2, !tbaa !0
  %v3 = insertelement <8 x i16> %v2, i16 %ld3, i64 3
  %gep4 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 2048
  %ld4 = load i16, ptr addrspace(5) %gep4, align 2, !tbaa !0
  %v4 = insertelement <8 x i16> %v3, i16 %ld4, i64 4
  %gep5 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 2560
  %ld5 = load i16, ptr addrspace(5) %gep5, align 2, !tbaa !0
  %v5 = insertelement <8 x i16> %v4, i16 %ld5, i64 5
  %gep6 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 3072
  %ld6 = load i16, ptr addrspace(5) %gep6, align 2, !tbaa !0
  %v6 = insertelement <8 x i16> %v5, i16 %ld6, i64 6
  %gep7 = getelementptr inbounds nuw i8, ptr addrspace(5) %gep, i32 3584
  %ld7 = load i16, ptr addrspace(5) %gep7, align 2, !tbaa !0
  %v7 = insertelement <8 x i16> %v6, i16 %ld7, i64 7
  %gep8 = getelementptr inbounds nuw [2 x i8], ptr addrspace(5) %ptr, i32 10
  %ld8 = load i16, ptr addrspace(5) %gep8, align 2, !tbaa !0
  %res = insertelement <8 x i16> %v6, i16 %ld8, i64 1
  ret <8 x i16> %res
}

define <8 x i16> @use_t16_flat(i32 %off, ptr %ptr) {
; GFX12-LABEL: name: use_t16_flat
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
; GFX12: FLAT_LOAD_SHORT_D16_t16 %{{[0-9]+}}
  %ext = zext i32 %off to i64
  %gep = getelementptr inbounds nuw [2 x i8], ptr %ptr, i64 %ext
  %ld0 = load i16, ptr %gep, align 2, !tbaa !0
  %v0 = insertelement <8 x i16> poison, i16 %ld0, i64 0
  %gep1 = getelementptr inbounds nuw i8, ptr %gep, i64 512
  %ld1 = load i16, ptr %gep1, align 2, !tbaa !0
  %v1 = insertelement <8 x i16> %v0, i16 %ld1, i64 1
  %gep2 = getelementptr inbounds nuw i8, ptr %gep, i64 1024
  %ld2 = load i16, ptr %gep2, align 2, !tbaa !0
  %v2 = insertelement <8 x i16> %v1, i16 %ld2, i64 2
  %gep3 = getelementptr inbounds nuw i8, ptr %gep, i64 1536
  %ld3 = load i16, ptr %gep3, align 2, !tbaa !0
  %v3 = insertelement <8 x i16> %v2, i16 %ld3, i64 3
  %gep4 = getelementptr inbounds nuw i8, ptr %gep, i64 2048
  %ld4 = load i16, ptr %gep4, align 2, !tbaa !0
  %v4 = insertelement <8 x i16> %v3, i16 %ld4, i64 4
  %gep5 = getelementptr inbounds nuw i8, ptr %gep, i64 2560
  %ld5 = load i16, ptr %gep5, align 2, !tbaa !0
  %v5 = insertelement <8 x i16> %v4, i16 %ld5, i64 5
  %gep6 = getelementptr inbounds nuw i8, ptr %gep, i64 3072
  %ld6 = load i16, ptr %gep6, align 2, !tbaa !0
  %v6 = insertelement <8 x i16> %v5, i16 %ld6, i64 6
  %gep7 = getelementptr inbounds nuw i8, ptr %gep, i64 3584
  %ld7 = load i16, ptr %gep7, align 2, !tbaa !0
  %v7 = insertelement <8 x i16> %v6, i16 %ld7, i64 7
  %gep8 = getelementptr inbounds nuw [2 x i8], ptr %ptr, i64 10
  %ld8 = load i16, ptr %gep8, align 2, !tbaa !0
  %res = insertelement <8 x i16> %v6, i16 %ld8, i64 1
  ret <8 x i16> %res
}

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C++ TBAA"}
