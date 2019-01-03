; RUN: llc -march=amdgcn -mcpu=gfx902 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}vector_clause:
; GCN:      global_load_dwordx4
; GCN-NEXT: global_load_dwordx4
; GCN-NEXT: global_load_dwordx4
; GCN-NEXT: global_load_dwordx4
; GCN-NEXT: s_nop
define amdgpu_kernel void @vector_clause(<4 x i32> addrspace(1)* noalias nocapture readonly %arg, <4 x i32> addrspace(1)* noalias nocapture %arg1) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = zext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp2
  %tmp4 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp3, align 16
  %tmp5 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp2
  %tmp6 = add nuw nsw i64 %tmp2, 1
  %tmp7 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp6
  %tmp8 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp7, align 16
  %tmp9 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp6
  %tmp10 = add nuw nsw i64 %tmp2, 2
  %tmp11 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp10
  %tmp12 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp11, align 16
  %tmp13 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp10
  %tmp14 = add nuw nsw i64 %tmp2, 3
  %tmp15 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp14
  %tmp16 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp15, align 16
  %tmp17 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp14
  store <4 x i32> %tmp4, <4 x i32> addrspace(1)* %tmp5, align 16
  store <4 x i32> %tmp8, <4 x i32> addrspace(1)* %tmp9, align 16
  store <4 x i32> %tmp12, <4 x i32> addrspace(1)* %tmp13, align 16
  store <4 x i32> %tmp16, <4 x i32> addrspace(1)* %tmp17, align 16
  ret void
}

; GCN-LABEL: {{^}}scalar_clause:
; GCN:      s_load_dwordx2
; GCN-NEXT: s_load_dwordx2
; GCN-NEXT: s_nop
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_load_dwordx4
; GCN-NEXT: s_load_dwordx4
; GCN-NEXT: s_load_dwordx4
; GCN-NEXT: s_load_dwordx4
define amdgpu_kernel void @scalar_clause(<4 x i32> addrspace(1)* noalias nocapture readonly %arg, <4 x i32> addrspace(1)* noalias nocapture %arg1) {
bb:
  %tmp = load <4 x i32>, <4 x i32> addrspace(1)* %arg, align 16
  %tmp2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 1
  %tmp3 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp2, align 16
  %tmp4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 1
  %tmp5 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 2
  %tmp6 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp5, align 16
  %tmp7 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 2
  %tmp8 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 3
  %tmp9 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp8, align 16
  %tmp10 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 3
  store <4 x i32> %tmp, <4 x i32> addrspace(1)* %arg1, align 16
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %tmp4, align 16
  store <4 x i32> %tmp6, <4 x i32> addrspace(1)* %tmp7, align 16
  store <4 x i32> %tmp9, <4 x i32> addrspace(1)* %tmp10, align 16
  ret void
}

; GCN-LABEL: {{^}}mubuf_clause:
; GCN:      buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: buffer_load_dword
; GCN-NEXT: s_nop
; GCN-NEXT: s_nop
; GCN-NEXT: buffer_load_dword
define void @mubuf_clause(<4 x i32> addrspace(5)* noalias nocapture readonly %arg, <4 x i32> addrspace(5)* noalias nocapture %arg1) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg, i32 %tmp
  %tmp3 = load <4 x i32>, <4 x i32> addrspace(5)* %tmp2, align 16
  %tmp4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg1, i32 %tmp
  %tmp5 = add nuw nsw i32 %tmp, 1
  %tmp6 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg, i32 %tmp5
  %tmp7 = load <4 x i32>, <4 x i32> addrspace(5)* %tmp6, align 16
  %tmp8 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg1, i32 %tmp5
  %tmp9 = add nuw nsw i32 %tmp, 2
  %tmp10 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg, i32 %tmp9
  %tmp11 = load <4 x i32>, <4 x i32> addrspace(5)* %tmp10, align 16
  %tmp12 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg1, i32 %tmp9
  %tmp13 = add nuw nsw i32 %tmp, 3
  %tmp14 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg, i32 %tmp13
  %tmp15 = load <4 x i32>, <4 x i32> addrspace(5)* %tmp14, align 16
  %tmp16 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %arg1, i32 %tmp13
  store <4 x i32> %tmp3, <4 x i32> addrspace(5)* %tmp4, align 16
  store <4 x i32> %tmp7, <4 x i32> addrspace(5)* %tmp8, align 16
  store <4 x i32> %tmp11, <4 x i32> addrspace(5)* %tmp12, align 16
  store <4 x i32> %tmp15, <4 x i32> addrspace(5)* %tmp16, align 16
  ret void
}

; GCN-LABEL: {{^}}vector_clause_indirect:
; GCN: global_load_dwordx2 [[ADDR:v\[[0-9:]+\]]], v[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-NEXT: s_nop 0
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_nop 0
; GCN-NEXT: global_load_dwordx4 v[{{[0-9:]+}}], [[ADDR]], off
; GCN-NEXT: global_load_dwordx4 v[{{[0-9:]+}}], [[ADDR]], off offset:16
define amdgpu_kernel void @vector_clause_indirect(i64 addrspace(1)* noalias nocapture readonly %arg, <4 x i32> addrspace(1)* noalias nocapture readnone %arg1, <4 x i32> addrspace(1)* noalias nocapture %arg2) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = zext i32 %tmp to i64
  %tmp4 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 %tmp3
  %tmp5 = bitcast i64 addrspace(1)* %tmp4 to <4 x i32> addrspace(1)* addrspace(1)*
  %tmp6 = load <4 x i32> addrspace(1)*, <4 x i32> addrspace(1)* addrspace(1)* %tmp5, align 8
  %tmp7 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp6, align 16
  %tmp8 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %tmp6, i64 1
  %tmp9 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp8, align 16
  store <4 x i32> %tmp7, <4 x i32> addrspace(1)* %arg2, align 16
  %tmp10 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg2, i64 1
  store <4 x i32> %tmp9, <4 x i32> addrspace(1)* %tmp10, align 16
  ret void
}

; GCN-LABEL: {{^}}load_global_d16_hi:
; GCN:      global_load_short_d16_hi v
; GCN-NEXT: s_nop
; GCN-NEXT: s_nop
; GCN-NEXT: global_load_short_d16_hi v
define void @load_global_d16_hi(i16 addrspace(1)* %in, i16 %reg, <2 x i16> addrspace(1)* %out) {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 32
  %load1 = load i16, i16 addrspace(1)* %in
  %load2 = load i16, i16 addrspace(1)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* %out
  %build2 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build3 = insertelement <2 x i16> %build2, i16 %load2, i32 1
  %gep2 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 32
  store <2 x i16> %build3, <2 x i16> addrspace(1)* %gep2
  ret void
}

; GCN-LABEL: {{^}}load_global_d16_lo:
; GCN:      global_load_short_d16 v
; GCN-NEXT: s_nop
; GCN-NEXT: s_nop
; GCN-NEXT: global_load_short_d16 v
define void @load_global_d16_lo(i16 addrspace(1)* %in, i32 %reg, <2 x i16> addrspace(1)* %out) {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 32
  %reg.bc1 = bitcast i32 %reg to <2 x i16>
  %reg.bc2 = bitcast i32 %reg to <2 x i16>
  %load1 = load i16, i16 addrspace(1)* %in
  %load2 = load i16, i16 addrspace(1)* %gep
  %build1 = insertelement <2 x i16> %reg.bc1, i16 %load1, i32 0
  %build2 = insertelement <2 x i16> %reg.bc2, i16 %load2, i32 0
  %gep2 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 32
  store <2 x i16> %build1, <2 x i16> addrspace(1)* %out
  store <2 x i16> %build2, <2 x i16> addrspace(1)* %gep2
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
