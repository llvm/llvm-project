; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=verde -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=verde -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=redwood < %s | FileCheck --check-prefixes=EG,FUNC %s
; RUN: opt -S -mtriple=amdgcn-- -passes='amdgpu-promote-alloca,sroa,instcombine' < %s | FileCheck -check-prefix=OPT %s
target datalayout = "A5"

; OPT-LABEL: @vector_read(
; OPT: %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, ptr addrspace(1) %out, align 4

; FUNC-LABEL: {{^}}vector_read:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
define amdgpu_kernel void @vector_read(ptr addrspace(1) %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 3
  store i32 0, ptr addrspace(5) %tmp
  store i32 1, ptr addrspace(5) %y
  store i32 2, ptr addrspace(5) %z
  store i32 3, ptr addrspace(5) %w
  %tmp1 = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %index
  %tmp2 = load i32, ptr addrspace(5) %tmp1
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @vector_write(
; OPT: %0 = insertelement <4 x i32> zeroinitializer, i32 1, i32 %w_index
; OPT: %1 = extractelement <4 x i32> %0, i32 %r_index
; OPT: store i32 %1, ptr addrspace(1) %out, align 4

; FUNC-LABEL: {{^}}vector_write:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
; EG: MOVA_INT
define amdgpu_kernel void @vector_write(ptr addrspace(1) %out, i32 %w_index, i32 %r_index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 3
  store i32 0, ptr addrspace(5) %tmp
  store i32 0, ptr addrspace(5) %y
  store i32 0, ptr addrspace(5) %z
  store i32 0, ptr addrspace(5) %w
  %tmp1 = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %w_index
  store i32 1, ptr addrspace(5) %tmp1
  %tmp2 = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %r_index
  %tmp3 = load i32, ptr addrspace(5) %tmp2
  store i32 %tmp3, ptr addrspace(1) %out
  ret void
}

; This test should be optimize to:
; store i32 0, ptr addrspace(1) %out

; OPT-LABEL: @bitcast_gep(
; OPT-LABEL: store i32 0, ptr addrspace(1) %out, align 4

; FUNC-LABEL: {{^}}bitcast_gep:
; EG: STORE_RAW
define amdgpu_kernel void @bitcast_gep(ptr addrspace(1) %out, i32 %w_index, i32 %r_index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 3
  store i32 0, ptr addrspace(5) %tmp
  store i32 0, ptr addrspace(5) %y
  store i32 0, ptr addrspace(5) %z
  store i32 0, ptr addrspace(5) %w
  %tmp1 = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %tmp4 = load i32, ptr addrspace(5) %tmp1
  store i32 %tmp4, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @vector_read_bitcast_gep(
; OPT: %0 = extractelement <4 x i32> <i32 1065353216, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, ptr addrspace(1) %out, align 4
define amdgpu_kernel void @vector_read_bitcast_gep(ptr addrspace(1) %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 3
  store float 1.0, ptr addrspace(5) %tmp
  store i32 1, ptr addrspace(5) %y
  store i32 2, ptr addrspace(5) %z
  store i32 3, ptr addrspace(5) %w
  %tmp1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %index
  %tmp2 = load i32, ptr addrspace(5) %tmp1
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @vector_read_bitcast_alloca(
; OPT: %0 = extractelement <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00>, i32 %index
; OPT: store float %0, ptr addrspace(1) %out, align 4
define amdgpu_kernel void @vector_read_bitcast_alloca(ptr addrspace(1) %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr inbounds [4 x float], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr inbounds [4 x float], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr inbounds [4 x float], ptr addrspace(5) %tmp, i32 0, i32 3
  store float 0.0, ptr addrspace(5) %tmp
  store float 1.0, ptr addrspace(5) %y
  store float 2.0, ptr addrspace(5) %z
  store float 4.0, ptr addrspace(5) %w
  %tmp1 = getelementptr inbounds [4 x float], ptr addrspace(5) %tmp, i32 0, i32 %index
  %tmp2 = load float, ptr addrspace(5) %tmp1
  store float %tmp2, ptr addrspace(1) %out
  ret void
}

; The pointer arguments in local address space should not affect promotion to vector.

; OPT-LABEL: @vector_read_with_local_arg(
; OPT: %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, ptr addrspace(1) %out, align 4
define amdgpu_kernel void @vector_read_with_local_arg(ptr addrspace(3) %stopper, ptr addrspace(1) %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %y = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 3
  store i32 0, ptr addrspace(5) %tmp
  store i32 1, ptr addrspace(5) %y
  store i32 2, ptr addrspace(5) %z
  store i32 3, ptr addrspace(5) %w
  %tmp1 = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %index
  %tmp2 = load i32, ptr addrspace(5) %tmp1
  store i32 %tmp2, ptr addrspace(1) %out
  ret void
}
