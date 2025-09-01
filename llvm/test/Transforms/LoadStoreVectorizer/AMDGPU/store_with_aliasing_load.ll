; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; Check that, in the presence of an aliasing load, the stores preceding the
; aliasing load are safe to vectorize.

; CHECK-LABEL: store_vectorize_with_alias
; CHECK: store <4 x float>
; CHECK: load <4 x float>
; CHECK: store <4 x float>

; Function Attrs: nounwind
define amdgpu_kernel void @store_vectorize_with_alias(ptr addrspace(1) %a, ptr addrspace(1) %b) #0 {
bb:
  %tmp1 = load float, ptr addrspace(1) %b, align 4

  store float %tmp1, ptr addrspace(1) %a, align 4
  %tmp3 = getelementptr i8, ptr addrspace(1) %a, i64 4
  store float %tmp1, ptr addrspace(1) %tmp3, align 4
  %tmp5 = getelementptr i8, ptr addrspace(1) %a, i64 8
  store float %tmp1, ptr addrspace(1) %tmp5, align 4
  %tmp7 = getelementptr i8, ptr addrspace(1) %a, i64 12
  store float %tmp1, ptr addrspace(1) %tmp7, align 4

  %tmp9 = getelementptr i8, ptr addrspace(1) %b, i64 16
  %tmp11 = load float, ptr addrspace(1) %tmp9, align 4
  %tmp12 = getelementptr i8, ptr addrspace(1) %b, i64 20
  %tmp14 = load float, ptr addrspace(1) %tmp12, align 4
  %tmp15 = getelementptr i8, ptr addrspace(1) %b, i64 24
  %tmp17 = load float, ptr addrspace(1) %tmp15, align 4
  %tmp18 = getelementptr i8, ptr addrspace(1) %b, i64 28
  %tmp20 = load float, ptr addrspace(1) %tmp18, align 4

  %tmp21 = getelementptr i8, ptr addrspace(1) %a, i64 16
  store float %tmp11, ptr addrspace(1) %tmp21, align 4
  %tmp23 = getelementptr i8, ptr addrspace(1) %a, i64 20
  store float %tmp14, ptr addrspace(1) %tmp23, align 4
  %tmp25 = getelementptr i8, ptr addrspace(1) %a, i64 24
  store float %tmp17, ptr addrspace(1) %tmp25, align 4
  %tmp27 = getelementptr i8, ptr addrspace(1) %a, i64 28
  store float %tmp20, ptr addrspace(1) %tmp27, align 4

  ret void
}

attributes #0 = { argmemonly nounwind }
