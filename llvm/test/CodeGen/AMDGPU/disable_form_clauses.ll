; RUN: llc -march=amdgcn -mcpu=gfx902 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr -stop-after=si-form-memory-clauses < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}name:{{[ 	]*}}vector_clause
; GCN:      BUNDLE
; GCN-NEXT: LOAD_DWORDX2
; GCN-NEXT: LOAD_DWORDX2
; GCN-NEXT: {{^ *[}]}}
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

; GCN-LABEL: {{^}}name:{{[ 	]*}}no_vector_clause
; GCN-NOT:   BUNDLE
define amdgpu_kernel void @no_vector_clause(<4 x i32> addrspace(1)* noalias nocapture readonly %arg, <4 x i32> addrspace(1)* noalias nocapture %arg1) #0 {
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

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { "amdgpu-max-memory-clause"="1" }

