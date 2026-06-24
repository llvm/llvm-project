; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
; RUN:     -amdgpu-disable-rewrite-mfma-form-sched-stage=false \
; RUN:     < %s | FileCheck %s
;
; Test: RewriteMFMAFormStage — v7_slice pattern (gfx950, mfma.f32.16x16x32.f16)
;
; Distilled from v7_slice.llir (Triton matmul kernel, gfx950).
;
; Key structural features preserved from v7_slice:
;   (1) Loop body (%loop) accumulates via loop-carried SCALAR float phis,
;       NOT <4 x float> phis. Each MFMA acc is built by insertelement from
;       4 individual float phis then used as src2/dst in MFMAs within the loop.
;   (2) Epilogue (%epilogue) ALSO contains MFMAs: the final K-tile iteration
;       is peeled. Epilogue MFMAs take scalar float phis from %loop as acc,
;       then produce results consumed by fptrunc+store — Case 2.
;   (3) Scalar float phis initialized to 0.0 in entry (non-MAI def) — Case 3.
;
; CFG: entry -> loop (back-edge) -> epilogue -> ret
;
; Pressure design (gfx950, 256 ArchVGPR limit):
;   16 loop-carried scalar float phis (4 chains x 4 elems) = 16 VGPRs.
;   4 MFMA chains of 2, each acc = 4 float phis = 16 VGPRs in loop.
;   4 x <8 x half> operands (a0, a1, b0, b1) = 16 VGPRs.
;   8 x <32 x float> loop-carried vec carriers (vc0-vc7) = 256 VGPRs.
;   Peak ArchVGPR = 256 + 16 + 16 + 16 = 304 > 256 -> RegionsWithExcessArchVGPR.
;   After rewrite: scalar phis=16 VGPR, <8 x half>=16 VGPR, vec=256 VGPR, MFMA acc->AGPR.
;   getRewriteCost() < 0 -> rewrite() fires.
;
; Expected:
;   Case 3: v_accvgpr_write inserted after insertelement defs of MFMA acc
;           (entry zeroinitializer -> scalar float phi -> insertelement = non-MAI)
;   Case 2: v_accvgpr_read before fptrunc in epilogue block
;   MFMA opcodes: v_mfma_f32_16x16x32_f16 vgprcd -> agprcd form

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @test_v7slice_scalar_phi_acc(
; CHECK-LABEL: test_v7slice_scalar_phi_acc:
; CHECK:       ; @test_v7slice_scalar_phi_acc
; Case 3: v_accvgpr_write inserted for scalar-phi acc (non-MAI def -> AGPR init).
; Loop body MFMAs use AGPR C/D (a[...], a[...], a[...], a[...]).
; CHECK:       ; %bb.0:                                ; %entry
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:       .LBB0_1:                                ; %loop
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}], a[{{[0-9:]+}}]
; Case 2: epilogue MFMAs take VGPR src0/src1, AGPR C/D. v_accvgpr_read before fptrunc.
; CHECK:       ; %bb.2:                                ; %epilogue
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_mfma_f32_16x16x32_f16 a[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    ptr addrspace(1) %out,
    <8 x half> %a0,
    <8 x half> %a1,
    <8 x half> %b0,
    <8 x half> %b1,
    i32 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; -------------------------------------------------------------------
  ; 8 x <32 x float> loop-carried vector carriers = 256 VGPRs.
  ; Self-insert with variable index prevents folding.
  ; Kept live by store use in loop body.
  ; -------------------------------------------------------------------
  %vc0 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc0n, %loop ]
  %vc1 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc1n, %loop ]
  %vc2 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc2n, %loop ]
  %vc3 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc3n, %loop ]
  %vc4 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc4n, %loop ]
  %vc5 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc5n, %loop ]
  %vc6 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc6n, %loop ]
  %vc7 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc7n, %loop ]

  ; -------------------------------------------------------------------
  ; 16 loop-carried scalar float phis = 16 VGPRs.
  ; These are the per-element accumulator slots, mirroring v7_slice's
  ; pattern where each output element is a separate loop-carried float.
  ; Initialized to 0.0 in entry (non-MAI def) -> Case 3 triggers on the
  ; insertelement that builds the MFMA src2 from these scalars.
  ; -------------------------------------------------------------------
  ; MFMA chain A acc (4 floats):
  %a0_0 = phi float [ 0.0, %entry ], [ %ra0_0, %loop ]
  %a0_1 = phi float [ 0.0, %entry ], [ %ra0_1, %loop ]
  %a0_2 = phi float [ 0.0, %entry ], [ %ra0_2, %loop ]
  %a0_3 = phi float [ 0.0, %entry ], [ %ra0_3, %loop ]
  ; MFMA chain B acc (4 floats):
  %a1_0 = phi float [ 0.0, %entry ], [ %ra1_0, %loop ]
  %a1_1 = phi float [ 0.0, %entry ], [ %ra1_1, %loop ]
  %a1_2 = phi float [ 0.0, %entry ], [ %ra1_2, %loop ]
  %a1_3 = phi float [ 0.0, %entry ], [ %ra1_3, %loop ]
  ; MFMA chain C acc (4 floats):
  %a2_0 = phi float [ 0.0, %entry ], [ %ra2_0, %loop ]
  %a2_1 = phi float [ 0.0, %entry ], [ %ra2_1, %loop ]
  %a2_2 = phi float [ 0.0, %entry ], [ %ra2_2, %loop ]
  %a2_3 = phi float [ 0.0, %entry ], [ %ra2_3, %loop ]
  ; MFMA chain D acc (4 floats):
  %a3_0 = phi float [ 0.0, %entry ], [ %ra3_0, %loop ]
  %a3_1 = phi float [ 0.0, %entry ], [ %ra3_1, %loop ]
  %a3_2 = phi float [ 0.0, %entry ], [ %ra3_2, %loop ]
  %a3_3 = phi float [ 0.0, %entry ], [ %ra3_3, %loop ]

  ; -------------------------------------------------------------------
  ; Build <4 x float> acc vectors from scalar phi elements (v7_slice pattern).
  ; Each insertelement is a non-MAI def of an element of the acc register.
  ; -------------------------------------------------------------------
  %accA0 = insertelement <4 x float> poison,  float %a0_0, i32 0
  %accA1 = insertelement <4 x float> %accA0, float %a0_1, i32 1
  %accA2 = insertelement <4 x float> %accA1, float %a0_2, i32 2
  %accA  = insertelement <4 x float> %accA2, float %a0_3, i32 3

  %accB0 = insertelement <4 x float> poison,  float %a1_0, i32 0
  %accB1 = insertelement <4 x float> %accB0, float %a1_1, i32 1
  %accB2 = insertelement <4 x float> %accB1, float %a1_2, i32 2
  %accB  = insertelement <4 x float> %accB2, float %a1_3, i32 3

  %accC0 = insertelement <4 x float> poison,  float %a2_0, i32 0
  %accC1 = insertelement <4 x float> %accC0, float %a2_1, i32 1
  %accC2 = insertelement <4 x float> %accC1, float %a2_2, i32 2
  %accC  = insertelement <4 x float> %accC2, float %a2_3, i32 3

  %accD0 = insertelement <4 x float> poison,  float %a3_0, i32 0
  %accD1 = insertelement <4 x float> %accD0, float %a3_1, i32 1
  %accD2 = insertelement <4 x float> %accD1, float %a3_2, i32 2
  %accD  = insertelement <4 x float> %accD2, float %a3_3, i32 3

  ; -------------------------------------------------------------------
  ; MFMA chains in loop body (2 MFMAs per chain, chained acc).
  ; Results are loop-carried back via extractelement -> scalar phi.
  ; -------------------------------------------------------------------
  %rA_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %accA, i32 0, i32 0, i32 0)
  %rA   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %rA_v, i32 0, i32 0, i32 0)

  %rB_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %accB, i32 0, i32 0, i32 0)
  %rB   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %rB_v, i32 0, i32 0, i32 0)

  %rC_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %accC, i32 0, i32 0, i32 0)
  %rC   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %rC_v, i32 0, i32 0, i32 0)

  %rD_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %accD, i32 0, i32 0, i32 0)
  %rD   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %rD_v, i32 0, i32 0, i32 0)

  ; Extract scalar results to carry back via phi.
  %ra0_0 = extractelement <4 x float> %rA, i32 0
  %ra0_1 = extractelement <4 x float> %rA, i32 1
  %ra0_2 = extractelement <4 x float> %rA, i32 2
  %ra0_3 = extractelement <4 x float> %rA, i32 3

  %ra1_0 = extractelement <4 x float> %rB, i32 0
  %ra1_1 = extractelement <4 x float> %rB, i32 1
  %ra1_2 = extractelement <4 x float> %rB, i32 2
  %ra1_3 = extractelement <4 x float> %rB, i32 3

  %ra2_0 = extractelement <4 x float> %rC, i32 0
  %ra2_1 = extractelement <4 x float> %rC, i32 1
  %ra2_2 = extractelement <4 x float> %rC, i32 2
  %ra2_3 = extractelement <4 x float> %rC, i32 3

  %ra3_0 = extractelement <4 x float> %rD, i32 0
  %ra3_1 = extractelement <4 x float> %rD, i32 1
  %ra3_2 = extractelement <4 x float> %rD, i32 2
  %ra3_3 = extractelement <4 x float> %rD, i32 3

  ; Vector carrier self-inserts (prevent register reuse).
  %eidx = and i32 %i, 31
  %vc0e = extractelement <32 x float> %vc0, i32 0
  %vc1e = extractelement <32 x float> %vc1, i32 0
  %vc2e = extractelement <32 x float> %vc2, i32 0
  %vc3e = extractelement <32 x float> %vc3, i32 0
  %vc4e = extractelement <32 x float> %vc4, i32 0
  %vc5e = extractelement <32 x float> %vc5, i32 0
  %vc6e = extractelement <32 x float> %vc6, i32 0
  %vc7e = extractelement <32 x float> %vc7, i32 0
  %vc0n = insertelement <32 x float> %vc0, float %vc0e, i32 %eidx
  %vc1n = insertelement <32 x float> %vc1, float %vc1e, i32 %eidx
  %vc2n = insertelement <32 x float> %vc2, float %vc2e, i32 %eidx
  %vc3n = insertelement <32 x float> %vc3, float %vc3e, i32 %eidx
  %vc4n = insertelement <32 x float> %vc4, float %vc4e, i32 %eidx
  %vc5n = insertelement <32 x float> %vc5, float %vc5e, i32 %eidx
  %vc6n = insertelement <32 x float> %vc6, float %vc6e, i32 %eidx
  %vc7n = insertelement <32 x float> %vc7, float %vc7e, i32 %eidx

  %vcs = fadd float %vc0e, %vc1e
  store float %vcs, ptr addrspace(1) %out, align 4

  %i.next = add i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %epilogue, label %loop

epilogue:
  ; -------------------------------------------------------------------
  ; v7_slice pattern: epilogue ALSO has MFMAs (peeled final K-tile).
  ; Acc built from the same scalar float phis coming out of %loop.
  ; These insertelements are non-MAI defs -> Case 3.
  ; -------------------------------------------------------------------
  %eaccA0 = insertelement <4 x float> poison,   float %ra0_0, i32 0
  %eaccA1 = insertelement <4 x float> %eaccA0, float %ra0_1, i32 1
  %eaccA2 = insertelement <4 x float> %eaccA1, float %ra0_2, i32 2
  %eaccA  = insertelement <4 x float> %eaccA2, float %ra0_3, i32 3

  %eaccB0 = insertelement <4 x float> poison,   float %ra1_0, i32 0
  %eaccB1 = insertelement <4 x float> %eaccB0, float %ra1_1, i32 1
  %eaccB2 = insertelement <4 x float> %eaccB1, float %ra1_2, i32 2
  %eaccB  = insertelement <4 x float> %eaccB2, float %ra1_3, i32 3

  %eaccC0 = insertelement <4 x float> poison,   float %ra2_0, i32 0
  %eaccC1 = insertelement <4 x float> %eaccC0, float %ra2_1, i32 1
  %eaccC2 = insertelement <4 x float> %eaccC1, float %ra2_2, i32 2
  %eaccC  = insertelement <4 x float> %eaccC2, float %ra2_3, i32 3

  %eaccD0 = insertelement <4 x float> poison,   float %ra3_0, i32 0
  %eaccD1 = insertelement <4 x float> %eaccD0, float %ra3_1, i32 1
  %eaccD2 = insertelement <4 x float> %eaccD1, float %ra3_2, i32 2
  %eaccD  = insertelement <4 x float> %eaccD2, float %ra3_3, i32 3

  ; Epilogue MFMAs (2 per chain, as in v7_slice).
  %erA_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %eaccA, i32 0, i32 0, i32 0)
  %erA   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %erA_v, i32 0, i32 0, i32 0)

  %erB_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %eaccB, i32 0, i32 0, i32 0)
  %erB   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %erB_v, i32 0, i32 0, i32 0)

  %erC_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %eaccC, i32 0, i32 0, i32 0)
  %erC   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %erC_v, i32 0, i32 0, i32 0)

  %erD_v = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a0, <8 x half> %b0, <4 x float> %eaccD, i32 0, i32 0, i32 0)
  %erD   = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a1, <8 x half> %b1, <4 x float> %erD_v, i32 0, i32 0, i32 0)

  ; -------------------------------------------------------------------
  ; Non-MAI use of epilogue MFMA results via fptrunc (Case 2).
  ; v7_slice: results shuffled and fptrunc'd to fp16 for stores.
  ; -------------------------------------------------------------------
  %shA = shufflevector <4 x float> %erA, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hA  = fptrunc <2 x float> %shA to <2 x half>
  store <2 x half> %hA, ptr addrspace(1) %out, align 2

  %shB = shufflevector <4 x float> %erB, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hB  = fptrunc <2 x float> %shB to <2 x half>
  %pB  = getelementptr i16, ptr addrspace(1) %out, i32 2
  store <2 x half> %hB, ptr addrspace(1) %pB, align 2

  %shC = shufflevector <4 x float> %erC, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hC  = fptrunc <2 x float> %shC to <2 x half>
  %pC  = getelementptr i16, ptr addrspace(1) %out, i32 4
  store <2 x half> %hC, ptr addrspace(1) %pC, align 2

  %shD = shufflevector <4 x float> %erD, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hD  = fptrunc <2 x float> %shD to <2 x half>
  %pD  = getelementptr i16, ptr addrspace(1) %out, i32 6
  store <2 x half> %hD, ptr addrspace(1) %pD, align 2

  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }
