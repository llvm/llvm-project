; RUN: llc -amdgpu-conditional-discard-transformations=1 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that the branch is removed by the discard opt.

; GCN-LABEL: {{^}}if_with_kill_true_cond:
; GCN:      v_cmp_ne_u32_e32 vcc,
; GCN-NEXT: s_and_b64 exec, exec, vcc
; GCN-NOT: branch
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
; GCN-NOT: branch
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

attributes #0 = { nounwind }

declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #0
declare void @llvm.amdgcn.kill(i1) #0

