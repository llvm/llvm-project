; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-disable-rewrite-mfma-form-sched-stage=false -stop-after=machine-scheduler < %s | FileCheck %s --check-prefix=AFTER
;
; Test: RewriteMFMAFormStage — self-referential MFMA with SubRanges from
; per-lane insertelement initialisation.  Validates the isSameInstr guard and
; the SubRange full-reg BFS path.
;
; After register coalescing, %acc and %r1 (MFMA_A dst) are the SAME VReg
; (self-referential: MFMA_A src2 == dst).  The entry path initialises %acc
; per-lane via insertelement → creates SubRanges on %acc's LiveInterval.
;
; CFG:
;
;   entry ──► loop ──► exit
;               ↑_____|
;            (back-edge)
;
; loop body (after coalescing, %acc = %r1 = same VReg):
;   %acc = phi [%acc_init, entry], [%acc, loop]   ; self-ref: MFMA_A writes %acc
;   %acc = MFMA_A(a, b, %acc)   src2==dst → candidate (dst only used by MFMA_B, MAI)
;   %r2  = MFMA_B(a, b, %acc)   dst → extractelement (non-MAI) → NOT candidate
;
; findReachingDefs for MFMA_A src2=%acc:
;   %acc has SubRanges (per-lane insertelement entry init) → full-reg use →
;   queries all SubRanges.  Each SubRange VNI at UseIdx is PHIDef → BFS entered.
;
;   BFS traversal (Visited={}):
;     BB#1(loop) end: SubRange VNI = MFMA_A def (self-ref: same instruction as use)
;                   → isSameInstr(MFMA_A_def, MFMA_A_UseIdx) = true → SKIP
;     BB#0(entry) end: SubRange VNI = subreg COPY (%acc.subN = COPY %acc_init.subN)
;                   → ResultSet.insert(COPY_def)  ← non-MAI
;
;   ResultSet = {COPY_def (non-MAI only)} → hasDominanceConflict = false
;   MFMA_A is a valid candidate; Cost = -220 < 0 → rewrite fires.
;   Bridge copy for %acc_init (VGPR→AGPR) inserted after entry subreg COPY.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @test_src2_subrange_nonmai_init(
    ptr addrspace(1) %out,
    <8 x half> %a,
    <8 x half> %b,
    float %f0, float %f1, float %f2, float %f3,
    i32 %n) #0 {
entry:
  ; Initialize accumulator lane-by-lane via insertelement (non-MAI).
  ; Creates per-lane SubRanges; no single SubRange covers the full operand.
  %i0 = insertelement <4 x float> undef, float %f0, i32 0
  %i1 = insertelement <4 x float> %i0,  float %f1, i32 1
  %i2 = insertelement <4 x float> %i1,  float %f2, i32 2
  %acc_init = insertelement <4 x float> %i2, float %f3, i32 3
  br label %loop

loop:
  %i    = phi i32         [ 0,              %entry ], [ %i.next, %loop ]
  %acc  = phi <4 x float> [ %acc_init,      %entry ], [ %r1,     %loop ]

  ; 8 × <32 x float> carriers = 256 ArchVGPRs (pressure for rewrite stage).
  %vc0  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc0n, %loop ]
  %vc1  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc1n, %loop ]
  %vc2  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc2n, %loop ]
  %vc3  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc3n, %loop ]
  %vc4  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc4n, %loop ]
  %vc5  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc5n, %loop ]
  %vc6  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc6n, %loop ]
  %vc7  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc7n, %loop ]

  ; MFMA_A: src2==dst (self-ref after coalescing).  isRewriteCandidate = true
  ; (dst only used by MFMA_B, MAI).  BFS skips self-ref back-edge def via
  ; isSameInstr; entry subreg COPY (non-MAI) is the only reaching def.
  ; hasDominanceConflict = false; Cost = -220 → rewritten to AGPR.
  %r1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %acc, i32 0, i32 0, i32 0)

  ; MFMA_B: dst used by extractelement (non-MAI) → isRewriteCandidate = false.
  %r2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %r1, i32 0, i32 0, i32 0)

  %eidx = and i32 %i, 31
  %v0e  = extractelement <32 x float> %vc0, i32 0
  %vc0n = insertelement <32 x float> %vc0, float %v0e, i32 %eidx
  %vc1n = insertelement <32 x float> %vc1, float %v0e, i32 %eidx
  %vc2n = insertelement <32 x float> %vc2, float %v0e, i32 %eidx
  %vc3n = insertelement <32 x float> %vc3, float %v0e, i32 %eidx
  %vc4n = insertelement <32 x float> %vc4, float %v0e, i32 %eidx
  %vc5n = insertelement <32 x float> %vc5, float %v0e, i32 %eidx
  %vc6n = insertelement <32 x float> %vc6, float %v0e, i32 %eidx
  %vc7n = insertelement <32 x float> %vc7, float %v0e, i32 %eidx
  store volatile float %v0e, ptr addrspace(1) %out, align 4

  %i.next = add i32 %i, 1
  %cond   = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  %e = extractelement <4 x float> %r2, i32 0
  store float %e, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }

; AFTER-LABEL: name: test_src2_subrange_nonmai_init
; AFTER: bb.1.loop:
; AFTER: %{{[0-9]+}}:areg_128_align2 = V_MFMA_F32_16X16X32_F16_e64
; AFTER: %{{[0-9]+}}:vreg_128_align2 = V_MFMA_F32_16X16X32_F16_vgprcd_e64
