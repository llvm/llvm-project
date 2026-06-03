; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-disable-rewrite-mfma-form-sched-stage=false -stop-after=machine-scheduler < %s | FileCheck %s --check-prefix=AFTER
;
; Test: RewriteMFMAFormStage — MFMA src2 sub-register overwrite creates SubRanges,
; forcing findReachingDefs to use the SubRanges path where BFS sees MFMA_C (MAI)
; as a direct reaching def on the back-edge.
;
; Sub-register overwrite mechanism:
;   %acc_init is built by four insertelement calls (one per lane) → per-lane
;   subreg defs.  After register coalescing, %acc and %r3 (MFMA_C dst) are the
;   SAME virtual register (%acc = %r3, coalesced).  But the entry path initialises
;   %acc per-lane via subreg COPYs (%acc.subN = COPY %acc_init.subN), while the
;   back-edge path is a full-register MFMA_C write.  This mixed subreg/full-reg
;   write pattern causes %acc's LiveInterval to have SubRanges: each SubRange's
;   VNI at UseIdx is PHIDef (the per-lane entry COPY and the full-reg MFMA_C
;   definition create a two-path join in each lane's SubRange).
;
; CFG:
;
;   entry ──► loop ──► bridge ──► (ret)
;     │          ↑_____|
;     │       (back-edge)
;     └──────────────────────────► bridge   (bypass: n==0 skips loop)
;
; loop body (after coalescing, %acc = %r3 = same VReg):
;   %acc  = phi [%acc_init, entry], [%acc, loop]  ; %acc written by MFMA_C each iter
;   %r1   = MFMA_A(a, b, %acc)   dst → MFMA_B (MAI only)  → candidate
;   %r2   = MFMA_B(a, b, %r1)   dst → MFMA_C (MAI only)  → candidate
;   %acc  = MFMA_C(a, b, %r2)   full-reg write of %acc (= %r3) → candidate
;
; bridge: %p = phi [%acc, loop], [%acc_init, entry]
;   Two predecessors → phi-elim emits separate subreg COPYs per predecessor,
;   not coalesced → visible in stop-before (bb.1 bypass and bb.2 preheader).
;
; findReachingDefs for MFMA_A src2=%acc:
;   %acc has SubRanges (subreg entry init) → full-reg use → queries all SubRanges.
;   Each SubRange: VNI at UseIdx is PHIDef → BFS entered with Visited={}.
;
;   BFS traversal:
;     bb.4(loop) end: SubRange VNI = MFMA_C def (full-reg write, opcode=MFMA)
;                   → isSameInstr(MFMA_C_def, MFMA_A_UseIdx) = false
;                   → ResultSet.insert(MFMA_C_def)  ← MAI
;     bb.2(preheader) end: SubRange VNI = subreg COPY def (%acc.subN = COPY %acc_init.subN)
;                   → ResultSet.insert(COPY_def)     ← non-MAI
;
;   hasDominanceConflict({MFMA_C(MAI), COPY(non-MAI)}):
;     preheader-BB dominates loop-BB → true → MFMA_A excluded.
;   Propagation: MFMA_B excluded (src2 = excluded MFMA_A dst),
;                MFMA_C excluded (src2 = excluded MFMA_B dst).
;   All three loop MFMAs stay in vreg form.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @test_src2_nonself_backedge(
    ptr addrspace(1) %out,
    <8 x half> %a,
    <8 x half> %b,
    float %f0, float %f1, float %f2, float %f3,
    i32 %n) #0 {
entry:
  %i0 = insertelement <4 x float> undef, float %f0, i32 0
  %i1 = insertelement <4 x float> %i0,  float %f1, i32 1
  %i2 = insertelement <4 x float> %i1,  float %f2, i32 2
  %acc_init = insertelement <4 x float> %i2,  float %f3, i32 3
  %bypass = icmp eq i32 %n, 0
  br i1 %bypass, label %bridge, label %loop

loop:
  %i    = phi i32         [ 0,         %entry ], [ %i.next, %loop ]
  %acc  = phi <4 x float> [ %acc_init, %entry ], [ %r3,     %loop ]
  %vc0  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc0n, %loop ]
  %vc1  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc1n, %loop ]
  %vc2  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc2n, %loop ]
  %vc3  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc3n, %loop ]
  %vc4  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc4n, %loop ]
  %vc5  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc5n, %loop ]
  %vc6  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc6n, %loop ]
  %vc7  = phi <32 x float> [ zeroinitializer, %entry ], [ %vc7n, %loop ]

  ; MFMA_A: src2=%acc (≠ dst=%r1).  dst → MFMA_B (MAI) → candidate.
  %r1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %acc, i32 0, i32 0, i32 0)

  ; MFMA_B: dst → MFMA_C (MAI) + bridge phi COPY → candidate.
  %r2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %r1, i32 0, i32 0, i32 0)

  ; MFMA_C: dst → loop back-edge phi COPY → candidate.
  %r3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %r2, i32 0, i32 0, i32 0)

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
  br i1 %cond, label %bridge, label %loop

bridge:
  %p = phi <4 x float> [ %r3, %loop ], [ %acc_init, %entry ]
  %bridge_mfma = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(
      <8 x half> %a, <8 x half> %b, <4 x float> %p, i32 0, i32 0, i32 0)
  %e1 = extractelement <4 x float> %bridge_mfma, i32 0
  %s0 = fadd float %e1, %e1
  store float %s0, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }

; AFTER-LABEL: name: test_src2_nonself_backedge
; MFMA_A excluded (src2 dominance conflict: entry non-MAI + MFMA_C MAI on back-edge).
; MFMA_B and MFMA_C excluded by forward propagation (src2 = excluded MFMA dst).
; All three loop MFMAs stay in vreg form.
; AFTER: bb.4.loop:
; AFTER-COUNT-3: %{{[0-9]+}}:vreg_128_align2 = V_MFMA_F32_16X16X32_F16_vgprcd_e64
