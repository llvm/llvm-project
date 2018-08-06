; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; Tests whether a load chain of 8 constants gets vectorized into a wider load.
; FUNC-LABEL: {{^}}constant_load_v8f32:
; GCN: s_load_dwordx8
; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @constant_load_v8f32(float addrspace(4)* noalias nocapture readonly %weights, float addrspace(1)* noalias nocapture %out_ptr) {
entry:
  %out_ptr.promoted = load float, float addrspace(1)* %out_ptr, align 4
  %tmp = load float, float addrspace(4)* %weights, align 4
  %add = fadd float %tmp, %out_ptr.promoted
  %arrayidx.1 = getelementptr inbounds float, float addrspace(4)* %weights, i64 1
  %tmp1 = load float, float addrspace(4)* %arrayidx.1, align 4
  %add.1 = fadd float %tmp1, %add
  %arrayidx.2 = getelementptr inbounds float, float addrspace(4)* %weights, i64 2
  %tmp2 = load float, float addrspace(4)* %arrayidx.2, align 4
  %add.2 = fadd float %tmp2, %add.1
  %arrayidx.3 = getelementptr inbounds float, float addrspace(4)* %weights, i64 3
  %tmp3 = load float, float addrspace(4)* %arrayidx.3, align 4
  %add.3 = fadd float %tmp3, %add.2
  %arrayidx.4 = getelementptr inbounds float, float addrspace(4)* %weights, i64 4
  %tmp4 = load float, float addrspace(4)* %arrayidx.4, align 4
  %add.4 = fadd float %tmp4, %add.3
  %arrayidx.5 = getelementptr inbounds float, float addrspace(4)* %weights, i64 5
  %tmp5 = load float, float addrspace(4)* %arrayidx.5, align 4
  %add.5 = fadd float %tmp5, %add.4
  %arrayidx.6 = getelementptr inbounds float, float addrspace(4)* %weights, i64 6
  %tmp6 = load float, float addrspace(4)* %arrayidx.6, align 4
  %add.6 = fadd float %tmp6, %add.5
  %arrayidx.7 = getelementptr inbounds float, float addrspace(4)* %weights, i64 7
  %tmp7 = load float, float addrspace(4)* %arrayidx.7, align 4
  %add.7 = fadd float %tmp7, %add.6
  store float %add.7, float addrspace(1)* %out_ptr, align 4
  ret void
}