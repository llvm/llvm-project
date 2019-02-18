; RUN: opt -mtriple=amdgcn-amd-amdhsa -basicaa -scoped-noalias -load-store-vectorizer -S -o - %s | FileCheck -check-prefix=SCOPE -check-prefix=ALL %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -basicaa -load-store-vectorizer -S -o - %s | FileCheck -check-prefix=NOSCOPE -check-prefix=ALL %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; This fails to vectorize if the !alias.scope is not used

; ALL-LABEL: @vectorize_alias_scope(
; SCOPE: load float, float addrspace(1)* %c
; SCOPE: bitcast float addrspace(1)* %a to <2 x float> addrspace(1)*
; SCOPE: store <2 x float> zeroinitializer
; SCOPE: store float %ld.c, float addrspace(1)* %b,

; NOSCOPE: store float
; NOSCOPE: load float
; NOSCOPE: store float
; NOSCOPE: store float
define amdgpu_kernel void @vectorize_alias_scope(float addrspace(1)* nocapture %a, float addrspace(1)* nocapture %b, float addrspace(1)* nocapture readonly %c) #0 {
entry:
  %a.idx.1 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  store float 0.0, float addrspace(1)* %a, align 4, !noalias !0
  %ld.c = load float, float addrspace(1)* %c, align 4, !alias.scope !0
  store float 0.0, float addrspace(1)* %a.idx.1, align 4, !noalias !0
  store float %ld.c, float addrspace(1)* %b, align 4, !noalias !0
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1}
!1 = distinct !{!1, !2, !"some scope"}
!2 = distinct !{!2, !"some domain"}
