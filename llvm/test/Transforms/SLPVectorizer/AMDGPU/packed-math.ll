; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -slp-vectorizer -dce < %s | FileCheck -check-prefixes=GCN,GFX9,GFX89 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -slp-vectorizer -dce < %s | FileCheck -check-prefixes=GCN,VI,GFX89 %s

; FIXME: Should still like to vectorize the memory operations for VI

; Simple 3-pair chain with loads and stores
; GCN-LABEL: @test1_as_3_3_3_v2f16(
; GFX89: load <2 x half>, <2 x half> addrspace(3)*
; GFX89: load <2 x half>, <2 x half> addrspace(3)*
; GFX89: fmul <2 x half>
; GFX89: store <2 x half> %{{.*}}, <2 x half> addrspace(3)* %
; GFX89: ret
define amdgpu_kernel void @test1_as_3_3_3_v2f16(half addrspace(3)* %a, half addrspace(3)* %b, half addrspace(3)* %c) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %i1 = load half, half addrspace(3)* %b, align 2
  %mul = fmul half %i0, %i1
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half addrspace(3)* %b, i64 1
  %i4 = load half, half addrspace(3)* %arrayidx4, align 2
  %mul5 = fmul half %i3, %i4
  store half %mul, half addrspace(3)* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  store half %mul5, half addrspace(3)* %arrayidx5, align 2
  ret void
}

; GCN-LABEL: @test1_as_3_0_0(
; GFX89: load <2 x half>, <2 x half> addrspace(3)*
; GFX89: load <2 x half>, <2 x half>*
; GFX89: fmul <2 x half>
; GFX89: store <2 x half> %{{.*}}, <2 x half>* %
; GFX89: ret
define amdgpu_kernel void @test1_as_3_0_0(half addrspace(3)* %a, half* %b, half* %c) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %i1 = load half, half* %b, align 2
  %mul = fmul half %i0, %i1
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half* %b, i64 1
  %i4 = load half, half* %arrayidx4, align 2
  %mul5 = fmul half %i3, %i4
  store half %mul, half* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half* %c, i64 1
  store half %mul5, half* %arrayidx5, align 2
  ret void
}

; GCN-LABEL: @test1_as_0_0_3_v2f16(
; GFX89: load <2 x half>, <2 x half>*
; GFX89: load <2 x half>, <2 x half>*
; GFX89: fmul <2 x half>
; GFX89: store <2 x half> %{{.*}}, <2 x half> addrspace(3)* %
; GFX89: ret
define amdgpu_kernel void @test1_as_0_0_3_v2f16(half* %a, half* %b, half addrspace(3)* %c) {
  %i0 = load half, half* %a, align 2
  %i1 = load half, half* %b, align 2
  %mul = fmul half %i0, %i1
  %arrayidx3 = getelementptr inbounds half, half* %a, i64 1
  %i3 = load half, half* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half* %b, i64 1
  %i4 = load half, half* %arrayidx4, align 2
  %mul5 = fmul half %i3, %i4
  store half %mul, half addrspace(3)* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  store half %mul5, half addrspace(3)* %arrayidx5, align 2
  ret void
}

; GCN-LABEL: @test1_fma_v2f16(
; GFX9: load <2 x half>
; GFX9: load <2 x half>
; GFX9: load <2 x half>
; GFX9: call <2 x half> @llvm.fma.v2f16(
; GFX9: store <2 x half>
define amdgpu_kernel void @test1_fma_v2f16(half addrspace(3)* %a, half addrspace(3)* %b, half addrspace(3)* %c, half addrspace(3)* %d) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %i1 = load half, half addrspace(3)* %b, align 2
  %i2 = load half, half addrspace(3)* %c, align 2
  %fma0 = call half @llvm.fma.f16(half %i0, half %i1, half %i2)
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half addrspace(3)* %b, i64 1
  %i4 = load half, half addrspace(3)* %arrayidx4, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  %i5 = load half, half addrspace(3)* %arrayidx5, align 2
  %fma1 = call half @llvm.fma.f16(half %i3, half %i4, half %i5)
  store half %fma0, half addrspace(3)* %d, align 2
  %arrayidx6 = getelementptr inbounds half, half addrspace(3)* %d, i64 1
  store half %fma1, half addrspace(3)* %arrayidx6, align 2
  ret void
}

; GCN-LABEL: @mul_scalar_v2f16(
; GFX9: load <2 x half>
; GFX9: fmul <2 x half>
; GFX9: store <2 x half>
define amdgpu_kernel void @mul_scalar_v2f16(half addrspace(3)* %a, half %scalar, half addrspace(3)* %c) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %mul = fmul half %i0, %scalar
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %mul5 = fmul half %i3, %scalar
  store half %mul, half addrspace(3)* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  store half %mul5, half addrspace(3)* %arrayidx5, align 2
  ret void
}

; GCN-LABEL: @fabs_v2f16
; GFX9: load <2 x half>
; GFX9: call <2 x half> @llvm.fabs.v2f16(
; GFX9: store <2 x half>
define amdgpu_kernel void @fabs_v2f16(half addrspace(3)* %a, half addrspace(3)* %c) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %fabs0 = call half @llvm.fabs.f16(half %i0)
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %fabs1 = call half @llvm.fabs.f16(half %i3)
  store half %fabs0, half addrspace(3)* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  store half %fabs1, half addrspace(3)* %arrayidx5, align 2
  ret void
}

; GCN-LABEL: @test1_fabs_fma_v2f16(
; GFX9: load <2 x half>
; GFX9: call <2 x half> @llvm.fabs.v2f16(
; GFX9: call <2 x half> @llvm.fma.v2f16(
; GFX9: store <2 x half>
define amdgpu_kernel void @test1_fabs_fma_v2f16(half addrspace(3)* %a, half addrspace(3)* %b, half addrspace(3)* %c, half addrspace(3)* %d) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %i1 = load half, half addrspace(3)* %b, align 2
  %i2 = load half, half addrspace(3)* %c, align 2
  %i0.fabs = call half @llvm.fabs.f16(half %i0)

  %fma0 = call half @llvm.fma.f16(half %i0.fabs, half %i1, half %i2)
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half addrspace(3)* %b, i64 1
  %i4 = load half, half addrspace(3)* %arrayidx4, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  %i5 = load half, half addrspace(3)* %arrayidx5, align 2
  %i3.fabs = call half @llvm.fabs.f16(half %i3)

  %fma1 = call half @llvm.fma.f16(half %i3.fabs, half %i4, half %i5)
  store half %fma0, half addrspace(3)* %d, align 2
  %arrayidx6 = getelementptr inbounds half, half addrspace(3)* %d, i64 1
  store half %fma1, half addrspace(3)* %arrayidx6, align 2
  ret void
}

; FIXME: Should do vector load and extract component for fabs
; GCN-LABEL: @test1_fabs_scalar_fma_v2f16(
; GFX9: load half
; GFX9: call half @llvm.fabs.f16(
; GFX9: load <2 x half>
; GFX9: load half
; GFX9: load <2 x half>
; GFX9: call <2 x half> @llvm.fma.v2f16(
; GFX9: store <2 x half>
define amdgpu_kernel void @test1_fabs_scalar_fma_v2f16(half addrspace(3)* %a, half addrspace(3)* %b, half addrspace(3)* %c, half addrspace(3)* %d) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %i1 = load half, half addrspace(3)* %b, align 2
  %i2 = load half, half addrspace(3)* %c, align 2
  %i1.fabs = call half @llvm.fabs.f16(half %i1)

  %fma0 = call half @llvm.fma.f16(half %i0, half %i1.fabs, half %i2)
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half addrspace(3)* %b, i64 1
  %i4 = load half, half addrspace(3)* %arrayidx4, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  %i5 = load half, half addrspace(3)* %arrayidx5, align 2
  %fma1 = call half @llvm.fma.f16(half %i3, half %i4, half %i5)
  store half %fma0, half addrspace(3)* %d, align 2
  %arrayidx6 = getelementptr inbounds half, half addrspace(3)* %d, i64 1
  store half %fma1, half addrspace(3)* %arrayidx6, align 2
  ret void
}

; GCN-LABEL: @canonicalize_v2f16
; GFX9: load <2 x half>
; GFX9: call <2 x half> @llvm.canonicalize.v2f16(
; GFX9: store <2 x half>
define amdgpu_kernel void @canonicalize_v2f16(half addrspace(3)* %a, half addrspace(3)* %c) {
  %i0 = load half, half addrspace(3)* %a, align 2
  %canonicalize0 = call half @llvm.canonicalize.f16(half %i0)
  %arrayidx3 = getelementptr inbounds half, half addrspace(3)* %a, i64 1
  %i3 = load half, half addrspace(3)* %arrayidx3, align 2
  %canonicalize1 = call half @llvm.canonicalize.f16(half %i3)
  store half %canonicalize0, half addrspace(3)* %c, align 2
  %arrayidx5 = getelementptr inbounds half, half addrspace(3)* %c, i64 1
  store half %canonicalize1, half addrspace(3)* %arrayidx5, align 2
  ret void
}

declare half @llvm.fabs.f16(half) #1
declare half @llvm.fma.f16(half, half, half) #1
declare half @llvm.canonicalize.f16(half) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
