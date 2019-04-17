; RUN: opt -mtriple=amdgcn-- -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

@lds = internal addrspace(3) global [512 x float] undef, align 4

; The original load has an implicit alignment of 4, and should not
; increase to an align 8 load.

; CHECK-LABEL: @load_keep_base_alignment_missing_align(
; CHECK: load <2 x float>, <2 x float> addrspace(3)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @load_keep_base_alignment_missing_align(float addrspace(1)* %out) {
  %ptr0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 11
  %val0 = load float, float addrspace(3)* %ptr0

  %ptr1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 12
  %val1 = load float, float addrspace(3)* %ptr1
  %add = fadd float %val0, %val1
  store float %add, float addrspace(1)* %out
  ret void
}


; CHECK-LABEL: @store_keep_base_alignment_missing_align(
; CHECK: store <2 x float> zeroinitializer, <2 x float> addrspace(3)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @store_keep_base_alignment_missing_align() {
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 1
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 2
  store float 0.0, float addrspace(3)* %arrayidx0
  store float 0.0, float addrspace(3)* %arrayidx1
  ret void
}
