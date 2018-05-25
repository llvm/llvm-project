; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_membound:
; MemoryBound: 1
; WaveLimiterHint : 1
define amdgpu_kernel void @test_membound(<4 x i32> addrspace(1)* nocapture readonly %arg, <4 x i32> addrspace(1)* nocapture %arg1) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = zext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp2
  %tmp4 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp3, align 16
  %tmp5 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp2
  store <4 x i32> %tmp4, <4 x i32> addrspace(1)* %tmp5, align 16
  %tmp6 = add nuw nsw i64 %tmp2, 1
  %tmp7 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp6
  %tmp8 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp7, align 16
  %tmp9 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp6
  store <4 x i32> %tmp8, <4 x i32> addrspace(1)* %tmp9, align 16
  %tmp10 = add nuw nsw i64 %tmp2, 2
  %tmp11 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp10
  %tmp12 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp11, align 16
  %tmp13 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp10
  store <4 x i32> %tmp12, <4 x i32> addrspace(1)* %tmp13, align 16
  %tmp14 = add nuw nsw i64 %tmp2, 3
  %tmp15 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i64 %tmp14
  %tmp16 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp15, align 16
  %tmp17 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg1, i64 %tmp14
  store <4 x i32> %tmp16, <4 x i32> addrspace(1)* %tmp17, align 16
  ret void
}

; GCN-LABEL: {{^}}test_large_stride:
; MemoryBound: 0
; WaveLimiterHint : 1
define amdgpu_kernel void @test_large_stride(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 4096
  %tmp1 = load i32, i32 addrspace(1)* %tmp, align 4
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  store i32 %tmp1, i32 addrspace(1)* %tmp2, align 4
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 8192
  %tmp4 = load i32, i32 addrspace(1)* %tmp3, align 4
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  store i32 %tmp4, i32 addrspace(1)* %tmp5, align 4
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 12288
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 3
  store i32 %tmp7, i32 addrspace(1)* %tmp8, align 4
  ret void
}

; GCN-LABEL: {{^}}test_indirect:
; MemoryBound: 0
; WaveLimiterHint : 1
define amdgpu_kernel void @test_indirect(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 3
  %tmp3 = bitcast i32 addrspace(1)* %arg to <4 x i32> addrspace(1)*
  %tmp4 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp3, align 4
  %tmp5 = extractelement <4 x i32> %tmp4, i32 0
  %tmp6 = sext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp6
  %tmp8 = load i32, i32 addrspace(1)* %tmp7, align 4
  store i32 %tmp8, i32 addrspace(1)* %arg, align 4
  %tmp9 = extractelement <4 x i32> %tmp4, i32 1
  %tmp10 = sext i32 %tmp9 to i64
  %tmp11 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp10
  %tmp12 = load i32, i32 addrspace(1)* %tmp11, align 4
  store i32 %tmp12, i32 addrspace(1)* %tmp, align 4
  %tmp13 = extractelement <4 x i32> %tmp4, i32 2
  %tmp14 = sext i32 %tmp13 to i64
  %tmp15 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp14
  %tmp16 = load i32, i32 addrspace(1)* %tmp15, align 4
  store i32 %tmp16, i32 addrspace(1)* %tmp1, align 4
  %tmp17 = extractelement <4 x i32> %tmp4, i32 3
  %tmp18 = sext i32 %tmp17 to i64
  %tmp19 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp18
  %tmp20 = load i32, i32 addrspace(1)* %tmp19, align 4
  store i32 %tmp20, i32 addrspace(1)* %tmp2, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
