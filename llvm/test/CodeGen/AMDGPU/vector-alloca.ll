; RUN: llc -march=amdgcn -mcpu=verde -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG -check-prefix=FUNC %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-promote-alloca -sroa -instcombine < %s | FileCheck -check-prefix=OPT %s

; OPT-LABEL: @vector_read(
; OPT: %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, i32 addrspace(1)* %out, align 4

; FUNC-LABEL: {{^}}vector_read:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
define amdgpu_kernel void @vector_read(i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 3
  store i32 0, i32* %x
  store i32 1, i32* %y
  store i32 2, i32* %z
  store i32 3, i32* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32* %tmp1
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_write(
; OPT: %0 = insertelement <4 x i32> zeroinitializer, i32 1, i32 %w_index
; OPT: %1 = extractelement <4 x i32> %0, i32 %r_index
; OPT: store i32 %1, i32 addrspace(1)* %out, align 4

; FUNC-LABEL: {{^}}vector_write:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
; EG: MOVA_INT
define amdgpu_kernel void @vector_write(i32 addrspace(1)* %out, i32 %w_index, i32 %r_index) {
entry:
  %tmp = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 3
  store i32 0, i32* %x
  store i32 0, i32* %y
  store i32 0, i32* %z
  store i32 0, i32* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 %w_index
  store i32 1, i32* %tmp1
  %tmp2 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 %r_index
  %tmp3 = load i32, i32* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; This test should be optimize to:
; store i32 0, i32 addrspace(1)* %out

; OPT-LABEL: @bitcast_gep(
; OPT-LABEL: store i32 0, i32 addrspace(1)* %out, align 4

; FUNC-LABEL: {{^}}bitcast_gep:
; EG: STORE_RAW
define amdgpu_kernel void @bitcast_gep(i32 addrspace(1)* %out, i32 %w_index, i32 %r_index) {
entry:
  %tmp = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 3
  store i32 0, i32* %x
  store i32 0, i32* %y
  store i32 0, i32* %z
  store i32 0, i32* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %tmp2 = bitcast i32* %tmp1 to [4 x i32]*
  %tmp3 = getelementptr [4 x i32], [4 x i32]* %tmp2, i32 0, i32 0
  %tmp4 = load i32, i32* %tmp3
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_read_bitcast_gep(
; OPT: %0 = extractelement <4 x i32> <i32 1065353216, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, i32 addrspace(1)* %out, align 4
define amdgpu_kernel void @vector_read_bitcast_gep(i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32]
  %x = getelementptr inbounds [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  %y = getelementptr inbounds [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %z = getelementptr inbounds [4 x i32], [4 x i32]* %tmp, i32 0, i32 2
  %w = getelementptr inbounds [4 x i32], [4 x i32]* %tmp, i32 0, i32 3
  %bc = bitcast i32* %x to float*
  store float 1.0, float* %bc
  store i32 1, i32* %y
  store i32 2, i32* %z
  store i32 3, i32* %w
  %tmp1 = getelementptr inbounds [4 x i32], [4 x i32]* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32* %tmp1
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; FIXME: Should be able to promote this. Instcombine should fold the
; cast in the hasOneUse case so it might not matter in practice

; OPT-LABEL: @vector_read_bitcast_alloca(
; OPT: alloca [4 x float]
; OPT: store float
; OPT: store float
; OPT: store float
; OPT: store float
; OPT: load float
define amdgpu_kernel void @vector_read_bitcast_alloca(float addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32]
  %tmp.bc = bitcast [4 x i32]* %tmp to [4 x float]*
  %x = getelementptr inbounds [4 x float], [4 x float]* %tmp.bc, i32 0, i32 0
  %y = getelementptr inbounds [4 x float], [4 x float]* %tmp.bc, i32 0, i32 1
  %z = getelementptr inbounds [4 x float], [4 x float]* %tmp.bc, i32 0, i32 2
  %w = getelementptr inbounds [4 x float], [4 x float]* %tmp.bc, i32 0, i32 3
  store float 0.0, float* %x
  store float 1.0, float* %y
  store float 2.0, float* %z
  store float 4.0, float* %w
  %tmp1 = getelementptr inbounds [4 x float], [4 x float]* %tmp.bc, i32 0, i32 %index
  %tmp2 = load float, float* %tmp1
  store float %tmp2, float addrspace(1)* %out
  ret void
}

; The pointer arguments in local address space should not affect promotion to vector.

; OPT-LABEL: @vector_read_with_local_arg(
; OPT: %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT: store i32 %0, i32 addrspace(1)* %out, align 4
define amdgpu_kernel void @vector_read_with_local_arg(i32 addrspace(3)* %stopper, i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 3
  store i32 0, i32* %x
  store i32 1, i32* %y
  store i32 2, i32* %z
  store i32 3, i32* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32* %tmp1
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}
