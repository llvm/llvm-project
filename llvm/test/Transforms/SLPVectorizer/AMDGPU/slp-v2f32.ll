; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX90A %s

; GCN-LABEL: @fadd_combine
; GFX908: fadd float
; GFX908: fadd float
; GFX90A: fadd <2 x float>
define amdgpu_kernel void @fadd_combine(float addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load float, float addrspace(1)* %tmp2, align 4
  %tmp4 = fadd float %tmp3, 1.000000e+00
  store float %tmp4, float addrspace(1)* %tmp2, align 4
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp5
  %tmp7 = load float, float addrspace(1)* %tmp6, align 4
  %tmp8 = fadd float %tmp7, 1.000000e+00
  store float %tmp8, float addrspace(1)* %tmp6, align 4
  ret void
}

; GCN-LABEL: @fmul_combine
; GFX908: fmul float
; GFX908: fmul float
; GFX90A: fmul <2 x float>
define amdgpu_kernel void @fmul_combine(float addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load float, float addrspace(1)* %tmp2, align 4
  %tmp4 = fmul float %tmp3, 1.000000e+00
  store float %tmp4, float addrspace(1)* %tmp2, align 4
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp5
  %tmp7 = load float, float addrspace(1)* %tmp6, align 4
  %tmp8 = fmul float %tmp7, 1.000000e+00
  store float %tmp8, float addrspace(1)* %tmp6, align 4
  ret void
}

; GCN-LABEL: @fma_combine
; GFX908: call float @llvm.fma.f32
; GFX908: call float @llvm.fma.f32
; GFX90A: call <2 x float> @llvm.fma.v2f32
define amdgpu_kernel void @fma_combine(float addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load float, float addrspace(1)* %tmp2, align 4
  %tmp4 = tail call float @llvm.fma.f32(float %tmp3, float 1.000000e+00, float 1.000000e+00)
  store float %tmp4, float addrspace(1)* %tmp2, align 4
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp5
  %tmp7 = load float, float addrspace(1)* %tmp6, align 4
  %tmp8 = tail call float @llvm.fma.f32(float %tmp7, float 1.000000e+00, float 1.000000e+00)
  store float %tmp8, float addrspace(1)* %tmp6, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.fma.f32(float, float, float)

