; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Check that, in the presence of an aliasing load, the stores preceding the
; aliasing load are safe to vectorize.

; CHECK-LABEL: store_vectorize_with_alias
; CHECK: store <4 x float>
; CHECK: load <4 x float>
; CHECK: store <4 x float>

; Function Attrs: nounwind
define amdgpu_kernel void @store_vectorize_with_alias(i8 addrspace(1)* %a, i8 addrspace(1)* %b) #0 {
bb:
  %tmp = bitcast i8 addrspace(1)* %b to float addrspace(1)*
  %tmp1 = load float, float addrspace(1)* %tmp, align 4

  %tmp2 = bitcast i8 addrspace(1)* %a to float addrspace(1)*
  store float %tmp1, float addrspace(1)* %tmp2, align 4
  %tmp3 = getelementptr i8, i8 addrspace(1)* %a, i64 4
  %tmp4 = bitcast i8 addrspace(1)* %tmp3 to float addrspace(1)*
  store float %tmp1, float addrspace(1)* %tmp4, align 4
  %tmp5 = getelementptr i8, i8 addrspace(1)* %a, i64 8
  %tmp6 = bitcast i8 addrspace(1)* %tmp5 to float addrspace(1)*
  store float %tmp1, float addrspace(1)* %tmp6, align 4
  %tmp7 = getelementptr i8, i8 addrspace(1)* %a, i64 12
  %tmp8 = bitcast i8 addrspace(1)* %tmp7 to float addrspace(1)*
  store float %tmp1, float addrspace(1)* %tmp8, align 4

  %tmp9 = getelementptr i8, i8 addrspace(1)* %b, i64 16
  %tmp10 = bitcast i8 addrspace(1)* %tmp9 to float addrspace(1)*
  %tmp11 = load float, float addrspace(1)* %tmp10, align 4
  %tmp12 = getelementptr i8, i8 addrspace(1)* %b, i64 20
  %tmp13 = bitcast i8 addrspace(1)* %tmp12 to float addrspace(1)*
  %tmp14 = load float, float addrspace(1)* %tmp13, align 4
  %tmp15 = getelementptr i8, i8 addrspace(1)* %b, i64 24
  %tmp16 = bitcast i8 addrspace(1)* %tmp15 to float addrspace(1)*
  %tmp17 = load float, float addrspace(1)* %tmp16, align 4
  %tmp18 = getelementptr i8, i8 addrspace(1)* %b, i64 28
  %tmp19 = bitcast i8 addrspace(1)* %tmp18 to float addrspace(1)*
  %tmp20 = load float, float addrspace(1)* %tmp19, align 4

  %tmp21 = getelementptr i8, i8 addrspace(1)* %a, i64 16
  %tmp22 = bitcast i8 addrspace(1)* %tmp21 to float addrspace(1)*
  store float %tmp11, float addrspace(1)* %tmp22, align 4
  %tmp23 = getelementptr i8, i8 addrspace(1)* %a, i64 20
  %tmp24 = bitcast i8 addrspace(1)* %tmp23 to float addrspace(1)*
  store float %tmp14, float addrspace(1)* %tmp24, align 4
  %tmp25 = getelementptr i8, i8 addrspace(1)* %a, i64 24
  %tmp26 = bitcast i8 addrspace(1)* %tmp25 to float addrspace(1)*
  store float %tmp17, float addrspace(1)* %tmp26, align 4
  %tmp27 = getelementptr i8, i8 addrspace(1)* %a, i64 28
  %tmp28 = bitcast i8 addrspace(1)* %tmp27 to float addrspace(1)*
  store float %tmp20, float addrspace(1)* %tmp28, align 4

  ret void
}

attributes #0 = { argmemonly nounwind }
