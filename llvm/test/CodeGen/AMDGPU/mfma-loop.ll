; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX908,GFX908_A %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX90A,GFX908_A,GFX940_A %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX940,GFX940_A %s

; GCN-LABEL: {{^}}test_mfma_loop_zeroinit:

; GCN-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}

; Check that we do not copy agprs to vgprs and back inside the loop.

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; Final result should be read only once after the loop.

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_zeroinit(ptr addrspace(1) %arg) #0 {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ zeroinitializer, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_unfoldable_splat:

; Check that we do not use 32 temp vgprs, but rotate 3 vgprs only.
; 3 vgprs are needed to avoid wait states between writes.
; Check that we do not use 32 temp sgprs as well.

; GCN:          v_mov_b32_e32 [[TMP:v[0-9]+]], 0x42f60000
; GCN-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_unfoldable_splat(ptr addrspace(1) %arg) #0 {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ <float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0, float 123.0>, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_non_splat:

; GCN-COUNT-31:    v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GCN:             v_accvgpr_write_b32 a{{[0-9]+}}, 1.0{{$}}

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_non_splat(ptr addrspace(1) %arg) #0 {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ <float 0.0, float 1.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_unfoldable_seq:

; Check that we do not use 32 temp vgprs, but rotate 3 vgprs only.
; 3 vgprs are needed to avoid wait states between writes.

; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x42f80000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x42fa0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x42fc0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x42fe0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43000000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43010000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43020000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43030000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43040000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43050000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43060000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43070000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43080000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43090000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430a0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430b0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430c0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430d0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430e0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x430f0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43100000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43110000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43120000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43130000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43140000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43150000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43160000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43170000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43180000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x43190000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0
; GFX-908: v_mov_b32_e32 v0, 0x431a0000
; GFX-908: s_nop 1
; GFX-908: v_accvgpr_write_b32 {{[0-9]+}}, v0

; FIXME: Constant is now in VGPR instead of SGPR.

; GFX940_A: v_mov_b32_e32 v{{[0-9]+}}, 0x4{{[0-9a-f]+}}
; GFX940_A-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_unfoldable_seq(ptr addrspace(1) %arg) #0 {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ <float 123.0, float 124.0, float 125.0, float 126.0, float 127.0, float 128.0, float 129.0, float 130.0, float 131.0, float 132.0, float 133.0, float 134.0, float 135.0, float 136.0, float 137.0, float 138.0, float 139.0, float 140.0, float 141.0, float 142.0, float 143.0, float 144.0, float 145.0, float 146.0, float 147.0, float 148.0, float 149.0, float 150.0, float 151.0, float 152.0, float 153.0, float 154.0>, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_vgpr_init:

; GCN-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, v0{{$}}

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_vgpr_init(ptr addrspace(1) %arg) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %init = bitcast i32 %tid to float
  %tmp0 = insertelement <32 x float> undef, float %init, i32 0
  %tmp1 = insertelement <32 x float> %tmp0, float %init, i32 1
  %tmp2 = insertelement <32 x float> %tmp1, float %init, i32 2
  %tmp3 = insertelement <32 x float> %tmp2, float %init, i32 3
  %tmp4 = insertelement <32 x float> %tmp3, float %init, i32 4
  %tmp5 = insertelement <32 x float> %tmp4, float %init, i32 5
  %tmp6 = insertelement <32 x float> %tmp5, float %init, i32 6
  %tmp7 = insertelement <32 x float> %tmp6, float %init, i32 7
  %tmp8 = insertelement <32 x float> %tmp7, float %init, i32 8
  %tmp9 = insertelement <32 x float> %tmp8, float %init, i32 9
  %tmp10 = insertelement <32 x float> %tmp9, float %init, i32 10
  %tmp11 = insertelement <32 x float> %tmp10, float %init, i32 11
  %tmp12 = insertelement <32 x float> %tmp11, float %init, i32 12
  %tmp13 = insertelement <32 x float> %tmp12, float %init, i32 13
  %tmp14 = insertelement <32 x float> %tmp13, float %init, i32 14
  %tmp15 = insertelement <32 x float> %tmp14, float %init, i32 15
  %tmp16 = insertelement <32 x float> %tmp15, float %init, i32 16
  %tmp17 = insertelement <32 x float> %tmp16, float %init, i32 17
  %tmp18 = insertelement <32 x float> %tmp17, float %init, i32 18
  %tmp19 = insertelement <32 x float> %tmp18, float %init, i32 19
  %tmp20 = insertelement <32 x float> %tmp19, float %init, i32 20
  %tmp21 = insertelement <32 x float> %tmp20, float %init, i32 21
  %tmp22 = insertelement <32 x float> %tmp21, float %init, i32 22
  %tmp23 = insertelement <32 x float> %tmp22, float %init, i32 23
  %tmp24 = insertelement <32 x float> %tmp23, float %init, i32 24
  %tmp25 = insertelement <32 x float> %tmp24, float %init, i32 25
  %tmp26 = insertelement <32 x float> %tmp25, float %init, i32 26
  %tmp27 = insertelement <32 x float> %tmp26, float %init, i32 27
  %tmp28 = insertelement <32 x float> %tmp27, float %init, i32 28
  %tmp29 = insertelement <32 x float> %tmp28, float %init, i32 29
  %tmp30 = insertelement <32 x float> %tmp29, float %init, i32 30
  %tmp31 = insertelement <32 x float> %tmp30, float %init, i32 31

  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %tmp31, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_sgpr_init:

; GFX908:          v_mov_b32_e32 [[TMP:v[0-9]+]], s{{[0-9]+}}
; GFX908-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GFX940_A-COUNT-32:        v_accvgpr_write_b32 [[LEAD:a[0-9]+]], s{{[0-9]+}}

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_sgpr_init(ptr addrspace(1) %arg, float %init) #0 {
entry:
  %tmp0 = insertelement <32 x float> undef, float %init, i32 0
  %tmp1 = insertelement <32 x float> %tmp0, float %init, i32 1
  %tmp2 = insertelement <32 x float> %tmp1, float %init, i32 2
  %tmp3 = insertelement <32 x float> %tmp2, float %init, i32 3
  %tmp4 = insertelement <32 x float> %tmp3, float %init, i32 4
  %tmp5 = insertelement <32 x float> %tmp4, float %init, i32 5
  %tmp6 = insertelement <32 x float> %tmp5, float %init, i32 6
  %tmp7 = insertelement <32 x float> %tmp6, float %init, i32 7
  %tmp8 = insertelement <32 x float> %tmp7, float %init, i32 8
  %tmp9 = insertelement <32 x float> %tmp8, float %init, i32 9
  %tmp10 = insertelement <32 x float> %tmp9, float %init, i32 10
  %tmp11 = insertelement <32 x float> %tmp10, float %init, i32 11
  %tmp12 = insertelement <32 x float> %tmp11, float %init, i32 12
  %tmp13 = insertelement <32 x float> %tmp12, float %init, i32 13
  %tmp14 = insertelement <32 x float> %tmp13, float %init, i32 14
  %tmp15 = insertelement <32 x float> %tmp14, float %init, i32 15
  %tmp16 = insertelement <32 x float> %tmp15, float %init, i32 16
  %tmp17 = insertelement <32 x float> %tmp16, float %init, i32 17
  %tmp18 = insertelement <32 x float> %tmp17, float %init, i32 18
  %tmp19 = insertelement <32 x float> %tmp18, float %init, i32 19
  %tmp20 = insertelement <32 x float> %tmp19, float %init, i32 20
  %tmp21 = insertelement <32 x float> %tmp20, float %init, i32 21
  %tmp22 = insertelement <32 x float> %tmp21, float %init, i32 22
  %tmp23 = insertelement <32 x float> %tmp22, float %init, i32 23
  %tmp24 = insertelement <32 x float> %tmp23, float %init, i32 24
  %tmp25 = insertelement <32 x float> %tmp24, float %init, i32 25
  %tmp26 = insertelement <32 x float> %tmp25, float %init, i32 26
  %tmp27 = insertelement <32 x float> %tmp26, float %init, i32 27
  %tmp28 = insertelement <32 x float> %tmp27, float %init, i32 28
  %tmp29 = insertelement <32 x float> %tmp28, float %init, i32 29
  %tmp30 = insertelement <32 x float> %tmp29, float %init, i32 30
  %tmp31 = insertelement <32 x float> %tmp30, float %init, i32 31

  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %tmp31, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_mixed_init:

; GCN-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v0
; GFX908-DAG:   v_mov_b32_e32 [[TMP:v[0-9]+]], s{{[0-9]+}}
; GFX940_A-DAG: s_load_dword [[TMP:s[0-9]+]],
; GCN-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX908-DAG:   v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}

; GFX90A-DAG:      v_accvgpr_write_b32 [[LEAD:a[0-9]+]], 0
; GFX90A-COUNT-28: v_accvgpr_write_b32 a{{[0-9]+}}, 0

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_mixed_init(ptr addrspace(1) %arg, float %x) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %init = bitcast i32 %tid to float
  %tmp0 = insertelement <32 x float> zeroinitializer, float %init, i32 0
  %tmp1 = insertelement <32 x float> %tmp0, float %x, i32 1

  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %tmp1, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_mfma_forward_init:

; GFX908-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908:          v_mfma_f32_32x32x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX90A-NOT:      v_accvgpr
; GFX90A:          v_mfma_f32_32x32x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}
; GFX90A-NOT:      v_accvgpr
; GFX940:          v_mfma_f32_32x32x1_2b_f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}
; GCN-NOT:         v_accvgpr

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_mfma_forward_init(ptr addrspace(1) %arg) #0 {
entry:
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> zeroinitializer, i32 0, i32 0, i32 0)

  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %mai.0, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_loop_agpr_init:

; GFX908-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908:          v_mfma_f32_32x32x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX90A-NOT:      v_accvgpr
; GFX90A:          v_mfma_f32_32x32x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}
; GFX90A-NOT:      v_accvgpr
; GFX940:          v_mfma_f32_32x32x1_2b_f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}

; Check that we are using only one tmp VGPR.

; GFX908:             v_accvgpr_read_b32 [[TMP:v[0-9]+]], a{{[0-9]+}}
; GFX940_A-COUNT-31:  v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}

; GCN: [[LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[LOOP]]

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_loop_agpr_init(ptr addrspace(1) %arg) #0 {
entry:
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %init = extractelement <32 x float> %mai.0, i32 0
  %tmp0 = insertelement <32 x float> undef, float %init, i32 0
  %tmp1 = insertelement <32 x float> %tmp0, float %init, i32 1
  %tmp2 = insertelement <32 x float> %tmp1, float %init, i32 2
  %tmp3 = insertelement <32 x float> %tmp2, float %init, i32 3
  %tmp4 = insertelement <32 x float> %tmp3, float %init, i32 4
  %tmp5 = insertelement <32 x float> %tmp4, float %init, i32 5
  %tmp6 = insertelement <32 x float> %tmp5, float %init, i32 6
  %tmp7 = insertelement <32 x float> %tmp6, float %init, i32 7
  %tmp8 = insertelement <32 x float> %tmp7, float %init, i32 8
  %tmp9 = insertelement <32 x float> %tmp8, float %init, i32 9
  %tmp10 = insertelement <32 x float> %tmp9, float %init, i32 10
  %tmp11 = insertelement <32 x float> %tmp10, float %init, i32 11
  %tmp12 = insertelement <32 x float> %tmp11, float %init, i32 12
  %tmp13 = insertelement <32 x float> %tmp12, float %init, i32 13
  %tmp14 = insertelement <32 x float> %tmp13, float %init, i32 14
  %tmp15 = insertelement <32 x float> %tmp14, float %init, i32 15
  %tmp16 = insertelement <32 x float> %tmp15, float %init, i32 16
  %tmp17 = insertelement <32 x float> %tmp16, float %init, i32 17
  %tmp18 = insertelement <32 x float> %tmp17, float %init, i32 18
  %tmp19 = insertelement <32 x float> %tmp18, float %init, i32 19
  %tmp20 = insertelement <32 x float> %tmp19, float %init, i32 20
  %tmp21 = insertelement <32 x float> %tmp20, float %init, i32 21
  %tmp22 = insertelement <32 x float> %tmp21, float %init, i32 22
  %tmp23 = insertelement <32 x float> %tmp22, float %init, i32 23
  %tmp24 = insertelement <32 x float> %tmp23, float %init, i32 24
  %tmp25 = insertelement <32 x float> %tmp24, float %init, i32 25
  %tmp26 = insertelement <32 x float> %tmp25, float %init, i32 26
  %tmp27 = insertelement <32 x float> %tmp26, float %init, i32 27
  %tmp28 = insertelement <32 x float> %tmp27, float %init, i32 28
  %tmp29 = insertelement <32 x float> %tmp28, float %init, i32 29
  %tmp30 = insertelement <32 x float> %tmp29, float %init, i32 30
  %tmp31 = insertelement <32 x float> %tmp30, float %init, i32 31

  br label %for.cond.preheader

for.cond.preheader:
  %phi = phi <32 x float> [ %tmp31, %entry ], [ %mai.1, %for.cond.preheader ]
  %c = phi i32 [ 0, %entry ], [ %inc, %for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_nested_loop_zeroinit:

; GFX908-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX90A:          v_accvgpr_write_b32 [[LEAD:a[0-9]+]], 0
; GFX90A-COUNT-31: v_accvgpr_mov_b32 a{{[0-9]+}}, [[LEAD]]

; Check that we do not copy agprs to vgprs and back in an outer loop.

; GCN: [[OUTER_LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GCN: [[INNER_LOOP:.LBB[0-9_]+]]:
; GCN-NOT:  v_accvgpr
; GFX908_A: v_mfma_f32_32x32x1f32
; GFX940:   v_mfma_f32_32x32x1_2b_f32
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[INNER_LOOP]]
; GCN-NOT:  v_accvgpr
; GCN:      s_cbranch_scc1 [[OUTER_LOOP]]

; Final result should be read only once after the loop.

; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX908-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]

define amdgpu_kernel void @test_mfma_nested_loop_zeroinit(ptr addrspace(1) %arg) #0 {
entry:
  br label %for.cond.preheader

for.cond.preheader:
  %phi.0 = phi <32 x float> [ zeroinitializer, %entry ], [ %mai.1, %inner.exit ]
  %c.0 = phi i32 [ 0, %entry ], [ %inc.0, %inner.exit ]
  br label %inner.for.cond.preheader

inner.for.cond.preheader:
  %phi = phi <32 x float> [ %phi.0, %for.cond.preheader ], [ %mai.1, %inner.for.cond.preheader ]
  %c = phi i32 [ 0, %for.cond.preheader ], [ %inc, %inner.for.cond.preheader ]
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %phi, i32 0, i32 0, i32 0)
  %inc = add nuw nsw i32 %c, 1
  %cc = icmp eq i32 %inc, 16
  br i1 %cc, label %inner.exit, label %inner.for.cond.preheader

inner.exit:
  %inc.0 = add nuw nsw i32 %c.0, 1
  %cc.0 = icmp eq i32 %inc.0, 16
  br i1 %cc.0, label %exit, label %for.cond.preheader

exit:
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
