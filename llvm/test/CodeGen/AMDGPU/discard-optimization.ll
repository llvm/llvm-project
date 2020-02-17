; RUN: llc -amdgpu-conditional-discard-transformations=1 --march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,KILL %s
; RUN: llc -amdgpu-conditional-discard-transformations=1 -amdgpu-transform-discard-to-demote --march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,DEMOTE %s

; Check that the branch is removed by the discard opt.

; GCN-LABEL: {{^}}if_with_kill_true_cond:
; GCN:      v_cmp_ne_u32_e32 vcc,
; GCN-NEXT: s_and_b64 exec, exec, vcc
define amdgpu_ps void @if_with_kill_true_cond(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  tail call void @llvm.amdgcn.kill(i1 false)
  br label %endif

endif:
  ret void
}

; Check that the branch is removed by the discard opt.

; GCN-LABEL: {{^}}if_with_kill_false_cond:
; GCN:      v_cmp_eq_u32_e32 vcc,
; GCN-NEXT: s_and_b64 exec, exec, vcc
define amdgpu_ps void @if_with_kill_false_cond(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %endif, label %then

then:
  tail call void @llvm.amdgcn.kill(i1 false)
  br label %endif

endif:
  ret void
}

; Check that the branch exiting the loop is a divergent one (s_cbranch_vccnz).
; This test exercises a loop with kill as the only exit.

; GCN-LABEL: {{^}}kill_with_loop_exit:
; GCN: s_cbranch_vccnz
; GCN: s_cbranch_vccnz
define amdgpu_ps void @kill_with_loop_exit(float inreg %inp0, float inreg %inp1, <4 x i32> inreg %inp2, float inreg %inp3) {
.entry:
  %tmp24 = fcmp olt float %inp0, 1.280000e+02
  %tmp25 = fcmp olt float %inp1, 1.280000e+02
  %tmp26 = and i1 %tmp24, %tmp25
  br i1 %tmp26, label %bb35, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.entry
  %tmp31 = fcmp ogt float %inp3, 0.0
  br label %bb

bb:                                               ; preds = %bb, %.preheader1.preheader
  %tmp30 = phi float [ %tmp32, %bb ], [ 1.500000e+00, %.preheader1.preheader ]
  %tmp32 = fadd reassoc nnan nsz arcp contract float %tmp30, 2.500000e-01
  %tmp34 = fadd reassoc nnan nsz arcp contract float %tmp30, 2.500000e-01
  br i1 %tmp31, label %bb, label %bb33

bb33:                                             ; preds = %bb
  call void @llvm.amdgcn.kill(i1 false)
  br label %bb35

bb35:                                             ; preds = %bb33, %.entry
  %tmp36 = phi float [ %tmp34, %bb33 ], [ 1.000000e+00, %.entry ]
  call void @llvm.amdgcn.exp.f32(i32 immarg 0, i32 immarg 15, float %tmp36, float %tmp36, float %tmp36, float %tmp36, i1 immarg true, i1 immarg true) #3
  ret void
}

; Check that the kill inside a loop is not optimized away.

; GCN-LABEL: {{^}}if_with_loop_kill_after:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]],
; GCN-NEXT: s_xor_b64 s[{{[0-9:]+}}], exec, [[SAVEEXEC]]
define amdgpu_ps void @if_with_loop_kill_after(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  br i1 %cc, label %loop, label %break

break:
  tail call void @llvm.amdgcn.kill(i1 false)
  br label %endif

endif:
  ret void
}

; Check that the kill inside a loop is not optimized away.

; GCN-LABEL: {{^}}if_with_kill_inside_loop:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]],
; GCN-NEXT: s_xor_b64 s[{{[0-9:]+}}], exec, [[SAVEEXEC]]
define amdgpu_ps void @if_with_kill_inside_loop(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  tail call void @llvm.amdgcn.kill(i1 false)
  br i1 %cc, label %loop, label %break

break:
  br label %endif

endif:
  ret void
}


; GCN-LABEL: {{^}}wqm_kill_to_demote1:
; GCN-NEXT: ; %.entry
; GCN: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN: s_wqm_b64 exec, exec
; DEMOTE: s_and_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], vcc
; GCN: image_sample
; GCN: v_add_f32_e32
; DEMOTE: s_and_b64 exec, exec, [[LIVE]]
; KILL: s_and_b64 exec, exec, [[ORIG]]
; GCN: image_sample
define amdgpu_ps <4 x float> @wqm_kill_to_demote1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
.entry:
  %z.cmp = fcmp olt float %z, 0.0
  br i1 %z.cmp, label %.continue, label %.kill

.kill:
  call void @llvm.amdgcn.kill(i1 false)
  br label %.export

.continue:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %tex1 = extractelement <4 x float> %tex, i32 0
  %coord1 = fadd float %tex0, %tex1
  %rtex.src = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord1, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  br label %.export

.export:
  %rtex = phi <4 x float> [ undef, %.kill ], [ %rtex.src, %.continue ]
  ret <4 x float> %rtex
}


; GCN-LABEL: {{^}}wqm_kill_to_demote2:
; GCN-NEXT: ; %.entry
; GCN: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN: s_wqm_b64 exec, exec
; GCN: image_sample
; DEMOTE: s_and_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], vcc
; GCN: v_add_f32_e32
; DEMOTE: s_and_b64 exec, exec, [[LIVE]]
; KILL: s_and_b64 exec, exec, [[ORIG]]
; GCN: image_sample
define amdgpu_ps <4 x float> @wqm_kill_to_demote2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
.entry:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %tex1 = extractelement <4 x float> %tex, i32 0
  %z.cmp = fcmp olt float %tex0, 0.0
  br i1 %z.cmp, label %.continue, label %.kill

.kill:
  call void @llvm.amdgcn.kill(i1 false)
  br label %.continue

.continue:
  %coord1 = fadd float %tex0, %tex1
  %rtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord1, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0

  ret <4 x float> %rtex
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #0
declare void @llvm.amdgcn.kill(i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

