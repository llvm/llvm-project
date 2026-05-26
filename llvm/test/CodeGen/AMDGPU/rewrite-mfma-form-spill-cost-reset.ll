; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
; RUN:     -amdgpu-disable-rewrite-mfma-form-sched-stage=false \
; RUN:     -verify-machineinstrs \
; RUN:     -stop-after=machine-scheduler \
; RUN:     < %s | FileCheck %s
;
; Regression test for resetRewriteCandsToVGPR() called on the SpillCost > 0
; early-return path inside getRewriteCost().
;
; Background
; ----------
; initHeuristics() speculatively rewrites MFMA candidates from VGPR form to
; AGPR form (setDesc + setRegClass) so that getRealRegPressure() can evaluate
; the post-rewrite spill cost. getRewriteCost() then loops over regions. If
; SpillCost > 0 in any region (rewriting increases spilling), it bails early.
;
; Before the fix, the early-return path did not call resetRewriteCandsToVGPR(),
; leaving MRI with AGPR register classes for registers that are still used as
; VGPR operands. The corrupted MRI state causes downstream passes to emit AGPR
; instructions even though the rewrite was supposed to be rejected.
;
; How SpillCost > 0 is triggered here
; ------------------------------------
; The "amdgpu-agpr-alloc"="0,0" attribute forces AGPRThreshold = 0 in
; getMaxNumVectorRegs(). After initHeuristics() reclassifies MFMA accumulators
; to AGPR, getRealRegPressure() reports all reclassified registers as AGPR
; "spills" (every AGPR exceeds the 0 threshold). For region 1 (the hot loop):
;   SpillCostAfter  = 76  (AGPR excess over AGPRThreshold=0)
;   SpillCostBefore = 72  (archVGPR excess before rewrite)
;   SpillCost = (76-72)*2*32 = 256 > 0  →  early bail fired.
;
; Expected behavior WITH the fix
; --------------------------------
; resetRewriteCandsToVGPR() is called before returning, restoring all
; candidate MFMA register classes and opcodes to VGPR form. rewrite() is
; never called. MachineVerifier passes cleanly.
;
; Expected behavior WITHOUT the fix
; ----------------------------------
; The early-return skips resetRewriteCandsToVGPR(). MRI retains AGPR classes
; and AGPR-form opcodes (V_MFMA_F32_16X16X32_F16_e64) for r1..r3, while r4
; (not a rewrite candidate) stays VGPR-form and uses r3 as src2. The
; MachineVerifier aborts: an areg_128_align2 register is used as src2 of a
; VGPR-form MFMA, violating the operand register-class constraint.
;
; CFG: entry → loop (back-edge) → epilogue → ret
;
; Chain structure (12 independent chains, depth-4):
;   acc_X  = phi [zeroinit, entry], [r1_X, loop]
;   r1_X   = mfma(a0, b0, acc_X)   ; level-1 loop-carried accumulator
;   r2_X   = mfma(a1, b1, r1_X)   ; level-2
;   r3_X   = mfma(a0, b0, r2_X)   ; level-3
;   r4_X   = mfma(a1, b1, r3_X)   ; level-4, dst escapes to epilogue (non-MAI)
;
; Pressure design (gfx950, 256 ArchVGPR limit):
;   8  x <32 x float> loop-carried vector carriers = 256 VGPRs
;   12 x r1_X loop-carried <4 x float>             =  48 VGPRs (always live)
;   12 x r4_X live-out to epilogue                 =  48 VGPRs (live at exit)
;   <8 x half> a0/b0 + a1/b1                       =  16 VGPRs
;   Loop ArchVGPR peak ~= 328 > 256  → RegionsWithExcessArchVGPR set

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @test_spill_cost_reset(
    ptr addrspace(1) %out,
    <8 x half> %a0, <8 x half> %a1,
    <8 x half> %b0, <8 x half> %b1,
    i32 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  ; 8 x <32 x float> loop-carried vector carriers = 256 VGPRs.
  %vc0  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc0n,  %loop ]
  %vc1  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc1n,  %loop ]
  %vc2  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc2n,  %loop ]
  %vc3  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc3n,  %loop ]
  %vc4  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc4n,  %loop ]
  %vc5  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc5n,  %loop ]
  %vc6  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc6n,  %loop ]
  %vc7  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc7n,  %loop ]
  ; 12 independent MFMA accumulators (loop-carried).
  %acc0  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_0,  %loop ]
  %acc1  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_1,  %loop ]
  %acc2  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_2,  %loop ]
  %acc3  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_3,  %loop ]
  %acc4  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_4,  %loop ]
  %acc5  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_5,  %loop ]
  %acc6  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_6,  %loop ]
  %acc7  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_7,  %loop ]
  %acc8  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_8,  %loop ]
  %acc9  = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_9,  %loop ]
  %acc10 = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_10, %loop ]
  %acc11 = phi <4 x float> [ zeroinitializer, %entry ], [ %r1_11, %loop ]
  ; Level-1
  %r1_0  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc0,  i32 0, i32 0, i32 0)
  %r1_1  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc1,  i32 0, i32 0, i32 0)
  %r1_2  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc2,  i32 0, i32 0, i32 0)
  %r1_3  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc3,  i32 0, i32 0, i32 0)
  %r1_4  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc4,  i32 0, i32 0, i32 0)
  %r1_5  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc5,  i32 0, i32 0, i32 0)
  %r1_6  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc6,  i32 0, i32 0, i32 0)
  %r1_7  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc7,  i32 0, i32 0, i32 0)
  %r1_8  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc8,  i32 0, i32 0, i32 0)
  %r1_9  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc9,  i32 0, i32 0, i32 0)
  %r1_10 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc10, i32 0, i32 0, i32 0)
  %r1_11 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %acc11, i32 0, i32 0, i32 0)
  ; Level-2
  %r2_0  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_0,  i32 0, i32 0, i32 0)
  %r2_1  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_1,  i32 0, i32 0, i32 0)
  %r2_2  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_2,  i32 0, i32 0, i32 0)
  %r2_3  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_3,  i32 0, i32 0, i32 0)
  %r2_4  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_4,  i32 0, i32 0, i32 0)
  %r2_5  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_5,  i32 0, i32 0, i32 0)
  %r2_6  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_6,  i32 0, i32 0, i32 0)
  %r2_7  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_7,  i32 0, i32 0, i32 0)
  %r2_8  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_8,  i32 0, i32 0, i32 0)
  %r2_9  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_9,  i32 0, i32 0, i32 0)
  %r2_10 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_10, i32 0, i32 0, i32 0)
  %r2_11 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r1_11, i32 0, i32 0, i32 0)
  ; Level-3
  %r3_0  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_0,  i32 0, i32 0, i32 0)
  %r3_1  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_1,  i32 0, i32 0, i32 0)
  %r3_2  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_2,  i32 0, i32 0, i32 0)
  %r3_3  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_3,  i32 0, i32 0, i32 0)
  %r3_4  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_4,  i32 0, i32 0, i32 0)
  %r3_5  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_5,  i32 0, i32 0, i32 0)
  %r3_6  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_6,  i32 0, i32 0, i32 0)
  %r3_7  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_7,  i32 0, i32 0, i32 0)
  %r3_8  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_8,  i32 0, i32 0, i32 0)
  %r3_9  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_9,  i32 0, i32 0, i32 0)
  %r3_10 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_10, i32 0, i32 0, i32 0)
  %r3_11 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %r2_11, i32 0, i32 0, i32 0)
  ; Level-4 — dst escapes to epilogue via extractelement (non-MAI user).
  %r4_0  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_0,  i32 0, i32 0, i32 0)
  %r4_1  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_1,  i32 0, i32 0, i32 0)
  %r4_2  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_2,  i32 0, i32 0, i32 0)
  %r4_3  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_3,  i32 0, i32 0, i32 0)
  %r4_4  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_4,  i32 0, i32 0, i32 0)
  %r4_5  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_5,  i32 0, i32 0, i32 0)
  %r4_6  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_6,  i32 0, i32 0, i32 0)
  %r4_7  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_7,  i32 0, i32 0, i32 0)
  %r4_8  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_8,  i32 0, i32 0, i32 0)
  %r4_9  = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_9,  i32 0, i32 0, i32 0)
  %r4_10 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_10, i32 0, i32 0, i32 0)
  %r4_11 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %r3_11, i32 0, i32 0, i32 0)
  ; Keep carriers live.
  %vc0n  = fadd <32 x float> %vc0,  %vc0
  %vc1n  = fadd <32 x float> %vc1,  %vc1
  %vc2n  = fadd <32 x float> %vc2,  %vc2
  %vc3n  = fadd <32 x float> %vc3,  %vc3
  %vc4n  = fadd <32 x float> %vc4,  %vc4
  %vc5n  = fadd <32 x float> %vc5,  %vc5
  %vc6n  = fadd <32 x float> %vc6,  %vc6
  %vc7n  = fadd <32 x float> %vc7,  %vc7
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %epilogue

epilogue:
  %e = extractelement <4 x float> %r4_0, i32 0
  store float %e, ptr addrspace(1) %out, align 4
  ret void
}

; "amdgpu-agpr-alloc"="0,0": forces AGPRThreshold=0 in getMaxNumVectorRegs(),
; which causes SpillCost > 0 after speculative AGPR reclassification in region 1.
; Without the fix, the SpillCost > 0 early-return path in getRewriteCost() skips
; resetRewriteCandsToVGPR(), leaving areg_128_align2 register classes in MRI and
; producing V_MFMA_F32_16X16X32_F16_e64 (AGPR form) in the loop — corrupted output.
; With the fix, resetRewriteCandsToVGPR() restores all candidates to vreg_128_align2
; and V_MFMA_F32_16X16X32_F16_vgprcd_e64 before the function returns Cost > 0.
attributes #0 = { "amdgpu-agpr-alloc"="0,0" "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1,1" }

; Verify the output is valid MIR. The -verify-machineinstrs flag above is the
; primary regression check: without the fix, the MachineVerifier aborts when it
; finds areg_128_align2 register classes on VGPR-form MFMA instructions left
; behind by the unguarded early-return path in getRewriteCost().
; CHECK: name: test_spill_cost_reset
