; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; FIXME: Should not vectorize on gfx8

; GCN-LABEL: @fadd_combine_v2f16
; GCN: fadd <2 x half>
define void @fadd_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = fadd half %tmp3, 1.000000e+00
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = fadd half %tmp7, 1.000000e+00
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should not vectorize on gfx8
; GCN-LABEL: @fsub_combine_v2f16
; GCN: fsub <2 x half>
define void @fsub_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = fsub half %tmp3, 1.000000e+00
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = fsub half %tmp7, 1.000000e+00
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should not vectorize on gfx8
; GCN-LABEL: @fmul_combine_v2f16
; GCN: fmul <2 x half>
define void @fmul_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = fmul half %tmp3, 1.000000e+00
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = fmul half %tmp7, 1.000000e+00
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @fdiv_combine_v2f16
; GCN: fdiv <2 x half>
define void @fdiv_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = fdiv half %tmp3, 1.000000e+00
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = fdiv half %tmp7, 1.000000e+00
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @frem_combine_v2f16
; GCN: frem <2 x half>
define void @frem_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = frem half %tmp3, 1.000000e+00
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = frem half %tmp7, 1.000000e+00
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should not vectorize on gfx8
; GCN-LABEL: @fma_combine_v2f16
; GCN: call <2 x half> @llvm.fma.v2f16
define amdgpu_kernel void @fma_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = tail call half @llvm.fma.f16(half %tmp3, half 1.000000e+00, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = tail call half @llvm.fma.f16(half %tmp7, half 1.000000e+00, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should not vectorize on gfx8
; GCN-LABEL: @fmuladd_combine_v2f16
; GCN: call <2 x half> @llvm.fmuladd.v2f16
define amdgpu_kernel void @fmuladd_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = tail call half @llvm.fmuladd.f16(half %tmp3, half 1.000000e+00, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = tail call half @llvm.fmuladd.f16(half %tmp7, half 1.000000e+00, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @minnum_combine_v2f16
; GFX8: call half @llvm.minnum.f16(
; GFX8: call half @llvm.minnum.f16(

; GFX9: call <2 x half> @llvm.minnum.v2f16
define void @minnum_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.minnum.f16(half %tmp3, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.minnum.f16(half %tmp7, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @maxnum_combine_v2f16
; GFX8: call half @llvm.maxnum.f16(
; GFX8: call half @llvm.maxnum.f16(

; GFX9: call <2 x half> @llvm.maxnum.v2f16
define void @maxnum_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.maxnum.f16(half %tmp3, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.maxnum.f16(half %tmp7, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should vectorize
; GCN-LABEL: @minimum_combine_v2f16
; GCN: call half @llvm.minimum.f16(
; GCN: call half @llvm.minimum.f16(
define void @minimum_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.minimum.f16(half %tmp3, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.minimum.f16(half %tmp7, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @maximum_combine_v2f16
; GCN: call half @llvm.maximum.f16(
; GCN: call half @llvm.maximum.f16(
define void @maximum_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.maximum.f16(half %tmp3, half 1.000000e+00)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.maximum.f16(half %tmp7, half 1.000000e+00)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @canonicalize_combine_v2f16
; GCN: call <2 x half> @llvm.canonicalize.v2f16(
define void @canonicalize_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.canonicalize.f16(half %tmp3)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.canonicalize.f16(half %tmp7)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @fabs_combine_v2f16
; GCN: call <2 x half> @llvm.fabs.v2f16(
define void @fabs_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.fabs.f16(half %tmp3)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.fabs.f16(half %tmp7)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @fneg_combine_v2f16
; GCN: fneg <2 x half>
define void @fneg_combine_v2f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = fneg half %tmp3
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = fneg half %tmp7
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; GCN-LABEL: @copysign_combine_v2f16
; GCN: call <2 x half> @llvm.copysign.v2f16(
define void @copysign_combine_v2f16(ptr addrspace(1) %arg, half %sign) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.copysign.f16(half %tmp3, half %sign)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.copysign.f16(half %tmp7, half %sign)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2
  ret void
}

; FIXME: Should always vectorize
; GCN-LABEL: @copysign_combine_v4f16
; GCN: call <2 x half> @llvm.copysign.v2f16(

; GFX8: call half @llvm.copysign.f16(
; GFX8: call half @llvm.copysign.f16(

; GFX9: call <2 x half> @llvm.copysign.v2f16(
define void @copysign_combine_v4f16(ptr addrspace(1) %arg, half %sign) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64

  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.copysign.f16(half %tmp3, half %sign)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2

  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.copysign.f16(half %tmp7, half %sign)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2

  %tmp9 = add nuw nsw i64 %tmp1, 2
  %tmp10 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp9
  %tmp11 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp12 = call half @llvm.copysign.f16(half %tmp11, half %sign)
  store half %tmp12, ptr addrspace(1) %tmp10, align 2

  %tmp13 = add nuw nsw i64 %tmp1, 3
  %tmp14 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp13
  %tmp15 = load half, ptr addrspace(1) %tmp14, align 2
  %tmp16 = call half @llvm.copysign.f16(half %tmp15, half %sign)
  store half %tmp16, ptr addrspace(1) %tmp14, align 2
  ret void
}

; GCN-LABEL: @canonicalize_combine_v4f16
; GCN: call <2 x half> @llvm.canonicalize.v2f16(
; GCN: call <2 x half> @llvm.canonicalize.v2f16(
define void @canonicalize_combine_v4f16(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64

  %tmp2 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load half, ptr addrspace(1) %tmp2, align 2
  %tmp4 = call half @llvm.canonicalize.f16(half %tmp3)
  store half %tmp4, ptr addrspace(1) %tmp2, align 2

  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp8 = call half @llvm.canonicalize.f16(half %tmp7)
  store half %tmp8, ptr addrspace(1) %tmp6, align 2

  %tmp9 = add nuw nsw i64 %tmp1, 2
  %tmp10 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp9
  %tmp11 = load half, ptr addrspace(1) %tmp6, align 2
  %tmp12 = call half @llvm.canonicalize.f16(half %tmp11)
  store half %tmp12, ptr addrspace(1) %tmp10, align 2

  %tmp13 = add nuw nsw i64 %tmp1, 3
  %tmp14 = getelementptr inbounds half, ptr addrspace(1) %arg, i64 %tmp13
  %tmp15 = load half, ptr addrspace(1) %tmp14, align 2
  %tmp16 = call half @llvm.canonicalize.f16(half %tmp15)
  store half %tmp16, ptr addrspace(1) %tmp14, align 2
  ret void
}
