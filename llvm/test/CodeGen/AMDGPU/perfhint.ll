; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_membound:
; GCN: MemoryBound: 1
; GCN: WaveLimiterHint : 1
define amdgpu_kernel void @test_membound(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = zext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg, i64 %tmp2
  %tmp4 = load <4 x i32>, ptr addrspace(1) %tmp3, align 16
  %tmp5 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg1, i64 %tmp2
  store <4 x i32> %tmp4, ptr addrspace(1) %tmp5, align 16
  %tmp6 = add nuw nsw i64 %tmp2, 1
  %tmp7 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg, i64 %tmp6
  %tmp8 = load <4 x i32>, ptr addrspace(1) %tmp7, align 16
  %tmp9 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg1, i64 %tmp6
  store <4 x i32> %tmp8, ptr addrspace(1) %tmp9, align 16
  ret void
}

; GCN-LABEL: {{^}}test_membound_1:
; GCN: MemoryBound: 1
define amdgpu_kernel void @test_membound_1(ptr addrspace(1) nocapture readonly %ptr.0,
                                           ptr addrspace(1) nocapture %ptr.1,
                                           <2 x double> %arg.0, i32 %arg.1, <4 x double> %arg.2) {
bb.entry:
  %id.32 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %id.0 = zext i32 %id.32 to i64
  %gep.0 = getelementptr inbounds <2 x double>, ptr addrspace(1) %ptr.0, i64 %id.0
  %ld.0 = load <2 x double>, ptr addrspace(1) %gep.0, align 16
  %add.0 = fadd <2 x double> %arg.0, %ld.0

  %id.1 = add nuw nsw i64 %id.0, 1
  %gep.1 = getelementptr inbounds <2 x double>, ptr addrspace(1) %ptr.0, i64 %id.1
  %ld.1 = load <2 x double>, ptr addrspace(1) %gep.1, align 16
  %add.1 = fadd <2 x double> %add.0, %ld.1

  %id.2 = add nuw nsw i64 %id.0, 2
  %gep.2 = getelementptr inbounds <2 x double>, ptr addrspace(1) %ptr.0, i64 %id.2
  %ld.2 = load <2 x double>, ptr addrspace(1) %gep.2, align 16
  %add.2 = fadd <2 x double> %add.1, %ld.2

  %id.3 = add nuw nsw i64 %id.0, 3
  %gep.3= getelementptr inbounds <2 x double>, ptr addrspace(1) %ptr.0, i64 %id.3
  %ld.3 = load <2 x double>, ptr addrspace(1) %gep.3, align 16
  %add.3 = fadd <2 x double> %add.2, %ld.3

  %id.4 = add nuw nsw i64 %id.0, 4
  %gep.4= getelementptr inbounds <2 x double>, ptr addrspace(1) %ptr.0, i64 %id.4
  %ld.4 = load <2 x double>, ptr addrspace(1) %gep.4, align 16
  %add.4 = fadd <2 x double> %add.3, %ld.4

  store <2 x double> %add.4, ptr addrspace(1) %ptr.1, align 16
  %cond = icmp eq i32 %arg.1, 0
  br i1 %cond, label %bb.true, label %bb.ret

bb.true:
  %i0.arg.0 = extractelement <2 x double> %arg.0, i32 0
  %i1.arg.0 = extractelement <2 x double> %arg.0, i32 1
  %add.1.0 = fadd double %i0.arg.0, %i1.arg.0
  %i0.arg.2 = extractelement <4 x double> %arg.2, i32 0
  %i1.arg.2 = extractelement <4 x double> %arg.2, i32 1
  %add.1.1 = fadd double %i0.arg.2, %i1.arg.2
  %add.1.2 = fadd double %add.1.0, %add.1.1
  %i2.arg.2 = extractelement <4 x double> %arg.2, i32 2
  %i3.arg.2 = extractelement <4 x double> %arg.2, i32 3
  %add.1.3 = fadd double %i2.arg.2, %i3.arg.2
  %add.1.4 = fadd double %add.1.2, %add.1.3
  %i0.add.0 = extractelement <2 x double> %add.0, i32 0
  %i1.add.0 = extractelement <2 x double> %add.0, i32 1
  %add.1.5 = fadd double %i0.add.0, %i1.add.0
  %add.1.6 = fadd double %add.1.4, %add.1.5
  %i0.add.1 = extractelement <2 x double> %add.1, i32 0
  %i1.add.1 = extractelement <2 x double> %add.1, i32 1
  %add.1.7 = fadd double %i0.add.1, %i1.add.1
  %add.1.8 = fadd double %add.1.6, %add.1.7
  %i0.add.2 = extractelement <2 x double> %add.2, i32 0
  %i1.add.2 = extractelement <2 x double> %add.2, i32 1
  %add.1.9 = fadd double %i0.add.2, %i1.add.2
  %add.1.10 = fadd double %add.1.8, %add.1.9

  store double %add.1.8, ptr addrspace(1) %ptr.1, align 8
  br label %bb.ret

bb.ret:
  ret void
}

; GCN-LABEL: {{^}}test_large_stride:
; GCN: MemoryBound: 0
; GCN: WaveLimiterHint : 1
define amdgpu_kernel void @test_large_stride(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 4096
  %tmp1 = load i32, ptr addrspace(1) %tmp, align 4
  %mul1 = mul i32 %tmp1, %tmp1
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  store i32 %mul1, ptr addrspace(1) %tmp2, align 4
  %tmp3 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 8192
  %tmp4 = load i32, ptr addrspace(1) %tmp3, align 4
  %mul4 = mul i32 %tmp4, %tmp4
  %tmp5 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 2
  store i32 %mul4, ptr addrspace(1) %tmp5, align 4
  %tmp6 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 12288
  %tmp7 = load i32, ptr addrspace(1) %tmp6, align 4
  %mul7 = mul i32 %tmp7, %tmp7
  %tmp8 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 3
  store i32 %mul7, ptr addrspace(1) %tmp8, align 4
  ret void
}

; GCN-LABEL: {{^}}test_indirect:
; GCN: MemoryBound: 1
; GCN: WaveLimiterHint : 1
define amdgpu_kernel void @test_indirect(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  %tmp1 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 2
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 3
  %tmp4 = load <4 x i32>, ptr addrspace(1) %arg, align 4
  %tmp5 = extractelement <4 x i32> %tmp4, i32 0
  %tmp6 = sext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tmp6
  %tmp8 = load i32, ptr addrspace(1) %tmp7, align 4
  store i32 %tmp8, ptr addrspace(1) %arg, align 4
  %tmp9 = extractelement <4 x i32> %tmp4, i32 1
  %tmp10 = sext i32 %tmp9 to i64
  %tmp11 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tmp10
  %tmp12 = load i32, ptr addrspace(1) %tmp11, align 4
  store i32 %tmp12, ptr addrspace(1) %tmp, align 4
  %tmp13 = extractelement <4 x i32> %tmp4, i32 2
  %tmp14 = sext i32 %tmp13 to i64
  %tmp15 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tmp14
  %tmp16 = load i32, ptr addrspace(1) %tmp15, align 4
  store i32 %tmp16, ptr addrspace(1) %tmp1, align 4
  %tmp17 = extractelement <4 x i32> %tmp4, i32 3
  %tmp18 = sext i32 %tmp17 to i64
  %tmp19 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tmp18
  %tmp20 = load i32, ptr addrspace(1) %tmp19, align 4
  store i32 %tmp20, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_indirect_through_phi:
; GCN: MemoryBound: 0
; GCN: WaveLimiterHint : 0
define amdgpu_kernel void @test_indirect_through_phi(ptr addrspace(1) %arg) {
bb:
  %load = load float, ptr addrspace(1) %arg, align 8
  %load.f = bitcast float %load to i32
  %n = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i32 [ %load.f, %bb ], [ %and2, %bb1 ]
  %ind = phi i32 [ 0, %bb ], [ %inc2, %bb1 ]
  %and1 = and i32 %phi, %n
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %and1
  store float %load, ptr addrspace(1) %gep, align 4
  %inc1 = add nsw i32 %phi, 1310720
  %and2 = and i32 %inc1, %n
  %inc2 = add nuw nsw i32 %ind, 1
  %cmp = icmp eq i32 %inc2, 1024
  br i1 %cmp, label %bb2, label %bb1

bb2:                                              ; preds = %bb1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
