; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa < %s \
; RUN:   | FileCheck %s --implicit-check-not='v_mov_b32_e32 v23,'
; RUN: llc -enable-new-pm -mtriple=amdgpu9.50-amd-amdhsa < %s \
; RUN:   | FileCheck %s --implicit-check-not='v_mov_b32_e32 v23,'

; The "amdgpu-num-vgpr"="24" attribute leaves v0..v23 addressable and the WWM
; allocator takes v23 for the SGPR spill lanes. AMDGPUReserveWWMRegs used to
; reserve it without refreshing the shared RegisterClassInfo, so
; AMDGPURewriteAGPRCopyMFMA was still offered v23 and its spill slot elimination
; assigned it: an assertion in VirtRegMap with assertions on, a silent clobber
; of the spilled SGPRs without.
;
; The implicit-check-not is what catches that clobber without assertions. The
; MFMA and ScratchSize checks stop the test passing vacuously if the rewrite
; stops running or the kernel stops spilling.

; CHECK-LABEL: {{^}}reserved_wwm_vgpr_not_in_alloc_order:
; CHECK: v_writelane_b32 v23,
; CHECK: v_mfma_f32_32x32x16_bf16 a[
; CHECK: v_readlane_b32 {{s[0-9]+}}, v23,
; CHECK: ; NumVgprs: 24
; CHECK: ; ScratchSize: {{[1-9][0-9]*}}
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #1
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #2
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #3

define amdgpu_kernel void @reserved_wwm_vgpr_not_in_alloc_order(i32 %arg6, i32 %arg11, i1 %arg15, i1 %arg16, i1 %arg21, ptr addrspace(3) %arg23, i1 %arg26, i1 %arg27, ptr addrspace(3) %arg31, ptr addrspace(3) %arg35, i32 %arg36, i1 %arg37, i1 %arg38, <4 x bfloat> %arg44, ptr addrspace(3) %arg46, <8 x bfloat> %phi123, <8 x bfloat> %bitcast214, i1 %phi119, <8 x i8> %bitcast252, <8 x bfloat> %sitofp, <8 x bfloat> %bitcast216, <8 x bfloat> %bitcast218, <8 x bfloat> %bitcast212) #4 {
bbl:
  br label %bbl97

bbl97:                                            ; preds = %bbl97, %bbl
  %phi = phi i1 [ false, %bbl ], [ %and171, %bbl97 ]
  %phi98 = phi i1 [ false, %bbl ], [ %and175, %bbl97 ]
  %phi99 = phi i1 [ false, %bbl ], [ %and179, %bbl97 ]
  %phi101 = phi i1 [ false, %bbl ], [ %arg15, %bbl97 ]
  %phi102 = phi i1 [ false, %bbl ], [ %arg26, %bbl97 ]
  %phi103 = phi i1 [ false, %bbl ], [ %arg37, %bbl97 ]
  %phi104 = phi i1 [ false, %bbl ], [ %arg16, %bbl97 ]
  %phi106 = phi i32 [ 0, %bbl ], [ 1, %bbl97 ]
  %phi1071 = phi i32 [ 0, %bbl ], [ 1, %bbl97 ]
  %phi112 = phi <4 x i32> [ zeroinitializer, %bbl ], [ %call184, %bbl97 ]
  %phi113 = phi <4 x i32> [ zeroinitializer, %bbl ], [ splat (i32 1), %bbl97 ]
  %phi116 = phi i1 [ false, %bbl ], [ %and250, %bbl97 ]
  %phi117 = phi i1 [ false, %bbl ], [ %and256, %bbl97 ]
  %phi118 = phi i1 [ false, %bbl ], [ %arg38, %bbl97 ]
  %phi121 = phi i32 [ 0, %bbl ], [ 1, %bbl97 ]
  %phi122 = phi <8 x bfloat> [ zeroinitializer, %bbl ], [ %sitofp272, %bbl97 ]
  %phi128 = phi <16 x float> [ zeroinitializer, %bbl ], [ %call281, %bbl97 ]
  %bitcast162 = bitcast <4 x i32> %phi112 to <8 x bfloat>
  %select163 = select i1 %arg21, <8 x bfloat> %bitcast162, <8 x bfloat> zeroinitializer
  %select = select i1 %phi, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  %getelementptr164 = getelementptr [2 x i8], ptr addrspace(3) null, i32 %phi1071
  store <8 x bfloat> %select, ptr addrspace(3) %getelementptr164, align 16
  %select159 = select i1 %phi98, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select159, ptr addrspace(3) null, align 16
  %bitcast160 = bitcast <4 x i32> %phi113 to <8 x bfloat>
  %select161 = select i1 %phi99, <8 x bfloat> %bitcast160, <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select161, ptr addrspace(3) %arg23, align 16
  store <8 x bfloat> %select163, ptr addrspace(3) null, align 16
  %call1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %icmp = icmp slt i32 %call1, 1
  %icmp170 = icmp slt i32 %phi106, 1
  %and171 = and i1 %icmp, %icmp170
  %icmp73 = icmp slt i32 %call1, %arg6
  %and175 = and i1 %icmp73, %icmp170
  %and = and i32 %call1, 1020
  %icmp75 = icmp slt i32 %and, 1
  %and179 = and i1 %icmp75, %icmp170
  %call184 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %select219 = select i1 %phi104, <8 x bfloat> %bitcast218, <8 x bfloat> zeroinitializer
  %select213 = select i1 %phi101, <8 x bfloat> %sitofp, <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select213, ptr addrspace(3) %arg23, align 16
  %select215 = select i1 %phi102, <8 x bfloat> %bitcast214, <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select215, ptr addrspace(3) null, align 16
  %select217 = select i1 %phi103, <8 x bfloat> %bitcast216, <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select217, ptr addrspace(3) %arg31, align 16
  store <8 x bfloat> %select219, ptr addrspace(3) %arg35, align 16
  %select227 = select i1 %phi118, <8 x bfloat> %phi123, <8 x bfloat> zeroinitializer
  %shufflevector230 = shufflevector <8 x bfloat> %select227, <8 x bfloat> zeroinitializer, <4 x i32> <i32 poison, i32 poison, i32 poison, i32 7>
  %select225 = select i1 %phi116, <8 x bfloat> %bitcast212, <8 x bfloat> zeroinitializer
  %select226 = select i1 %phi117, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  %shufflevector229 = shufflevector <8 x bfloat> %select225, <8 x bfloat> %select226, <4 x i32> <i32 7, i32 15, i32 poison, i32 poison>
  %shufflevector231 = shufflevector <4 x bfloat> %shufflevector229, <4 x bfloat> %shufflevector230, <4 x i32> <i32 0, i32 1, i32 7, i32 poison>
  %select228 = select i1 %arg27, <8 x bfloat> %phi122, <8 x bfloat> zeroinitializer
  %shufflevector232 = shufflevector <8 x bfloat> %select228, <8 x bfloat> zeroinitializer, <4 x i32> <i32 poison, i32 poison, i32 poison, i32 7>
  %shufflevector233 = shufflevector <4 x bfloat> %shufflevector231, <4 x bfloat> %shufflevector232, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  store <4 x bfloat> %shufflevector233, ptr addrspace(3) null, align 8
  %icmp249 = icmp slt i32 %phi121, %arg6
  %and250 = and i1 %phi119, %icmp249
  %and64 = and i32 %call1, 1
  %icmp783 = icmp slt i32 %and64, 1
  %icmp255 = icmp slt i32 %phi121, %arg36
  %and256 = and i1 %icmp783, %icmp255
  %sitofp272 = sitofp <8 x i8> %bitcast252 to <8 x bfloat>
  %call153 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer, <16 x float> %phi128, i32 0, i32 0, i32 0)
  %call281 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> %call153, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  br i1 %arg15, label %bbl97, label %bbl290

bbl290:                                           ; preds = %bbl97
  %select292 = select i1 %and171, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select292, ptr addrspace(3) null, align 16
  %select294 = select i1 %and175, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select294, ptr addrspace(3) %arg46, align 16
  %select296 = select i1 %and179, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  store <8 x bfloat> %select296, ptr addrspace(3) null, align 16
  %getelementptr301 = getelementptr [2 x i8], ptr addrspace(3) null, i32 %arg11
  store <8 x bfloat> zeroinitializer, ptr addrspace(3) %getelementptr301, align 16
  %select314 = select i1 %and256, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  %shufflevector316 = shufflevector <8 x bfloat> zeroinitializer, <8 x bfloat> %select314, <4 x i32> <i32 0, i32 8, i32 poison, i32 poison>
  %shufflevector318 = shufflevector <4 x bfloat> %shufflevector316, <4 x bfloat> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %shufflevector319 = shufflevector <4 x bfloat> %shufflevector318, <4 x bfloat> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { "amdgpu-num-vgpr"="24" }
