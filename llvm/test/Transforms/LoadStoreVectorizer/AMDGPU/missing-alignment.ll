; RUN: opt -mtriple=amdgcn-- -load-store-vectorizer -S -o - %s | FileCheck %s

@lds = internal addrspace(3) global [512 x float] undef, align 4

; The original load has an implicit alignment of 4, and should not
; increase to an align 8 load.

; CHECK-LABEL: @load_keep_base_alignment_missing_align(
; CHECK: load <2 x float>, <2 x float> addrspace(3)* %{{[0-9]+}}, align 4
define void @load_keep_base_alignment_missing_align(float addrspace(1)* %out) {
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
define void @store_keep_base_alignment_missing_align() {
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 1
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 2
  store float 0.0, float addrspace(3)* %arrayidx0
  store float 0.0, float addrspace(3)* %arrayidx1
  ret void
}
