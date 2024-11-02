; RUN: llc < %s | FileCheck %s
; RUN: %if ptxas %{ llc < %s | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: test1
; CHECK: ld.global.v2.f32
; CHECK: ld.global.v2.f32
; CHECK: st.global.v2.f32
; CHECK: st.global.v2.f32
define void @test1(float addrspace(1)* noalias align 8 %in, float addrspace(1)* noalias align 8 %out) {
  %in.1 = getelementptr float, float addrspace(1)* %in, i32 1
  %in.2 = getelementptr float, float addrspace(1)* %in, i32 2
  %in.3 = getelementptr float, float addrspace(1)* %in, i32 3
  %v0 = load float, float addrspace(1)* %in, align 8
  %v1 = load float, float addrspace(1)* %in.1, align 4
  %v2 = load float, float addrspace(1)* %in.2, align 8
  %v3 = load float, float addrspace(1)* %in.3, align 4
  %sum0 = fadd float %v0, %v1
  %sum1 = fadd float %v1, %v2
  %sum2 = fadd float %v3, %v1
  %sum3 = fadd float %v2, %v3
  %out.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.3 = getelementptr float, float addrspace(1)* %out, i32 3
  store float %sum0, float addrspace(1)* %out, align 8
  store float %sum1, float addrspace(1)* %out.1, align 4
  store float %sum2, float addrspace(1)* %out.2, align 8
  store float %sum3, float addrspace(1)* %out.3, align 4
  ret void
}
