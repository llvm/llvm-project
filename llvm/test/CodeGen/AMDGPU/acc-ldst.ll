; RUN: llc -mtriple=amdgcn -mcpu=gfx90a < %s | FileCheck -enable-var-scope --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -early-live-intervals < %s | FileCheck -enable-var-scope --check-prefix=GCN %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL:  {{^}}test_load_mfma_store16:
; GCN-COUNT-8: global_load_dwordx4 a[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:     v_accvgpr_write
; GCN:         v_mfma_f32_32x32x1f32
; GCN-NEXT:    s_nop 7
; GCN-NEXT:    s_nop 7
; GCN-NEXT:    s_nop 2
; GCN-NOT:     v_accvgpr_read
; GCN-COUNT-8: global_store_dwordx4 v{{[0-9:]+}}, a[{{[0-9:]+}}], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_mfma_store16(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load1_mfma_store1:
; GCN:      global_load_dword a{{[0-9]+}}, v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:  v_accvgpr_read
; GCN:      v_mfma_f32_32x32x1f32 a[[[N:[0-9]+]]:
; GCN-NEXT: s_nop 7
; GCN-NEXT: s_nop 7
; GCN-NEXT: s_nop 2
; GCN-NOT:  v_accvgpr_read
; GCN-NEXT: global_store_dword v{{[0-9:]+}}, a[[N]], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_load1_mfma_store1(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load float, ptr addrspace(1) %gep
  %init = insertelement <32 x float> zeroinitializer, float %in.1, i32 0
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %init, i32 1, i32 2, i32 3)
  %elt = extractelement <32 x float> %mai.1, i32 0
  store float %elt, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load4_mfma_store4:
; GCN:      global_load_dwordx4 a[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:  v_accvgpr_write
; GCN:      v_mfma_i32_4x4x4i8 [[A:a\[[0-9:]+\]]]
; GCN-NEXT: s_nop 4
; GCN-NOT:  v_accvgpr_read
; GCN-NEXT: global_store_dwordx4 v{{[0-9:]+}}, [[A]], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_load4_mfma_store4(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <4 x i32>, ptr addrspace(1) %gep
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %in.1, i32 0, i32 0, i32 0)
  store <4 x i32> %mai.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load_store:
; GCN-COUNT-8: global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:     v_accvgpr
; GCN-COUNT-8: global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.1 = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %gep.2 = getelementptr inbounds <32 x float>, ptr addrspace(1) %gep.1, i32 32
  %in.1 = load <32 x float>, ptr addrspace(1) %gep.1
  store <32 x float> %in.1, ptr addrspace(1) %gep.2
  ret void
}

; GCN-LABEL: {{^}}test_load_add_mfma_store:
; GCN-COUNT-8:  global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-COUNT-32: v_accvgpr_write
; GCN:          v_mfma_f32_32x32x1f32
; GCN-NEXT:     s_nop 7
; GCN-NEXT:     s_nop 7
; GCN-NEXT:     s_nop 2
; GCN-NOT:      v_accvgpr_read
; GCN-COUNT-8:  global_store_dwordx4 v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_add_mfma_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %add.1 = fadd <32 x float> %in.1, %in.1
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %add.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load_add_store:
; GCN-COUNT-8:  global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:      v_accvgpr
; GCN-COUNT-16: v_pk_add_f32
; GCN-NOT:      v_accvgpr
; GCN-COUNT-8:  global_store_dwordx4 v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_add_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %add.1 = fadd <32 x float> %in.1, %in.1
  store <32 x float> %add.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load_mfma_add_store:
; GCN-COUNT-8:  global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-COUNT-32: v_accvgpr_write
; GCN:          v_mfma_f32_32x32x1f32
; GCN-COUNT-32: v_accvgpr_read
; GCN:          v_pk_add_f32
; GCN-COUNT-8:  global_store_dwordx4 v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_mfma_add_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  %add.1 = fadd <32 x float> %mai.1, %in.1
  store <32 x float> %add.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_load_add_mfma_mul_store:
; GCN-COUNT-8:  global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN:          v_pk_add_f32
; GCN-COUNT-32: v_accvgpr_write
; GCN:          v_mfma_f32_32x32x1f32
; GCN-COUNT-32: v_accvgpr_read
; GCN:          v_pk_mul_f32
; GCN-COUNT-8:  global_store_dwordx4 v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_load_add_mfma_mul_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %add.1 = fadd <32 x float> %in.1, %in.1
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %add.1, i32 1, i32 2, i32 3)
  %mul.1 = fmul <32 x float> %mai.1, %mai.1
  store <32 x float> %mul.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_mixeduse_load_add_mfma_mul_store:
; GCN-COUNT-8:  global_load_dwordx4 v[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-COUNT-32: v_accvgpr_write
; GCN:          v_mfma_f32_32x32x1f32
; GCN-COUNT-32: v_accvgpr_read
; GCN:          v_pk_mul_f32
; GCN-COUNT-8:  global_store_dwordx4 v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_mixeduse_load_add_mfma_mul_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep
  %add.1 = fadd <32 x float> %in.1, %in.1
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %add.1, i32 1, i32 2, i32 3)
  %mul.1 = fmul <32 x float> %mai.1, %in.1
  store <32 x float> %mul.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_multiuse_load_mfma_mfma_store:
; GCN-COUNT-8: global_load_dwordx4 a[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:     v_accvgpr_write
; GCN:         v_mfma_f32_32x32x1f32
; GCN-NOT:     v_accvgpr_read
; GCN-COUNT-8: global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
define amdgpu_kernel void @test_multiuse_load_mfma_mfma_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.1 = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %gep.2 = getelementptr inbounds <32 x float>, ptr addrspace(1) %gep.1, i32 32
  %in.1 = load <32 x float>, ptr addrspace(1) %gep.1
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, ptr addrspace(1) %gep.1
  store <32 x float> %mai.2, ptr addrspace(1) %gep.2
  ret void
}

; NB: for atomics both vdata and vdst shall be either VGPR or AGPR
; GCN-LABEL: {{^}}test_atomic_mfma_4xi32_atomic_store:
; GCN: v_accvgpr_write_b32 [[A_ZERO:a[0-9]+]], 0
; GCN:     global_atomic_sub [[IN:v[0-9]+]], v{{[0-9:]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] glc
; GCN-DAG: v_accvgpr_mov_b32 a{{[0-9]+}}, [[A_ZERO]]
; GCN-DAG: v_accvgpr_mov_b32 a{{[0-9]+}}, [[A_ZERO]]
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, [[IN]]
; GCN:     v_mfma_i32_4x4x4i8 a[[[N:[0-9]+]]:
; GCN:     v_accvgpr_read_b32 [[V:v[0-9]+]], a[[N]]{{$}}
; GCN:     global_atomic_add v{{[0-9]+}}, v{{[0-9:]+}}, [[V]], s[{{[0-9:]+}}] glc
; GCN:     global_store_dword v{{[0-9]+}}, v{{[0-9]+}},
define amdgpu_kernel void @test_atomic_mfma_4xi32_atomic_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %tid
  %in.1 = atomicrmw volatile sub ptr addrspace(1) %gep, i32 1 syncscope("agent") seq_cst, !amdgpu.no.remote.memory !0
  %tmp0 = insertelement <4 x i32> poison, i32 %in.1, i32 0
  %tmp1 = insertelement <4 x i32> %tmp0, i32 0, i32 1
  %tmp2 = insertelement <4 x i32> %tmp1, i32 0, i32 2
  %tmp3 = insertelement <4 x i32> %tmp2, i32 0, i32 3
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %tmp3, i32 0, i32 0, i32 0)
  %elt = extractelement <4 x i32> %mai.1, i32 0
  %val = atomicrmw volatile add ptr addrspace(1) %gep, i32 %elt syncscope("agent") seq_cst
  store i32 %val, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_atomic_mfma_4xi32_atomic64_store:
; GCN:         global_atomic_sub_x2 v[{{[0-9:]+}}], v{{[0-9:]+}}, v[{{[0-9:]+}}], s[{{[0-9:]+}}] glc
; GCN: v_accvgpr_write_b32 [[A_ZERO:a[0-9]+]], 0
; GCN: v_accvgpr_mov_b32 a{{[0-9]+}}, [[A_ZERO]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:         v_mfma_i32_4x4x4i8 a[[[N:[0-9]+]]:
; GCN:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; GCN:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; GCN:         global_atomic_add_x2 v[{{[0-9:]+}}], v{{[0-9:]+}}, v[{{[0-9:]+}}], s[{{[0-9:]+}}] glc
define amdgpu_kernel void @test_atomic_mfma_4xi32_atomic64_store(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %tid
  %in.1 = atomicrmw volatile sub ptr addrspace(1) %gep, i64 1 syncscope("agent") seq_cst, !amdgpu.no.remote.memory !0
  %tmp0 = insertelement <2 x i64> poison, i64 %in.1, i32 0
  %tmp1 = insertelement <2 x i64> %tmp0, i64 0, i32 1
  %tmp2 = bitcast <2 x i64> %tmp1 to <4 x i32>
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %tmp2, i32 0, i32 0, i32 0)
  %elt.1 = extractelement <4 x i32> %mai.1, i32 0
  %elt.2 = extractelement <4 x i32> %mai.1, i32 1
  %v2.1 = insertelement <2 x i32> poison, i32 %elt.1, i32 0
  %v2.2 = insertelement <2 x i32> %v2.1, i32 %elt.2, i32 1
  %v2 = bitcast <2 x i32> %v2.2 to i64
  %val = atomicrmw volatile add ptr addrspace(1) %gep, i64 %v2 syncscope("agent") seq_cst
  store i64 %val, ptr addrspace(1) %arg
  ret void
}

; NB: both data operands should be VGPR or AGPR
; GCN-LABEL: {{^}}test_load_mfma_ds2_store:
; GCN-DAG: ds_read_b128 [[IN:a\[[0-9:]+\]]], v{{[0-9:]+}}
; GCN-NOT: v_accvgpr_write
; GCN-DAG: v_mfma_i32_4x4x4i8 a[[[N:[0-9]+]]:{{[0-9]+}}], v{{[0-9:]+}}, v{{[0-9:]+}}, [[IN]]
; GCN-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NOT: v_accvgpr_read
; GCN:     ds_write_b32 v{{[0-9]+}}, a[[N]] offset:128
define amdgpu_kernel void @test_load_mfma_ds2_store(ptr addrspace(3) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.1 = getelementptr inbounds <4 x i32>, ptr addrspace(3) %arg, i32 %tid
  %in.1 = load <4 x i32>, ptr addrspace(3) %gep.1
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %in.1, i32 0, i32 0, i32 0)
  %elt = extractelement <4 x i32> %mai.1, i32 0
  %gep.2 = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 32
  store i32 1, ptr addrspace(3) %arg
  store i32 %elt, ptr addrspace(3) %gep.2
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_4xi32:
; GCN:     global_load_dwordx4 [[IN:a\[[0-9:]+\]]], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT: v_accvgpr_write
; GCN:     v_mfma_i32_4x4x4i8 [[RES:a\[[0-9:]+\]]], v{{[0-9:]+}}, v{{[0-9:]+}}, [[IN]]
; GCN-NOT: v_accvgpr_read
; GCN:     global_store_dwordx4 v[{{[0-9:]+}}], [[RES]],
define amdgpu_kernel void @test_mfma_loop_4xi32(ptr addrspace(1) %arg) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg, i32 %tid
  %in = load <4 x i32>, ptr addrspace(1) %gep
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <4 x i32> [ %in, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <4 x i32> %mai.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_32xfloat:
; GCN-COUNT-8: global_load_dwordx4 a[{{[0-9:]+}}], v{{[0-9:]+}}, s[{{[0-9:]+}}]
; GCN-NOT:     v_accvgpr_write
; GCN:         v_mfma_f32_32x32x1f32
; GCN-NOT:     v_accvgpr_read
; GCN-COUNT-8: global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}],
; GCN:         s_endpgm
define amdgpu_kernel void @test_mfma_loop_32xfloat(ptr addrspace(1) %arg) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in = load <32 x float>, ptr addrspace(1) %gep
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %in, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %gep
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }

!0 = !{}
