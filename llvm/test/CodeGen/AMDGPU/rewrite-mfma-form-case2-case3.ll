; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a \
; RUN:     -amdgpu-disable-rewrite-mfma-form-sched-stage=false \
; RUN:     < %s | FileCheck %s

define amdgpu_kernel void @test_case2_case3(
; CHECK-LABEL: test_case2_case3:
; CHECK:       ; @test_case2_case3
; Case 3: v_accvgpr_write in entry. Loop MFMAs all-AGPR. Case 2: v_accvgpr_read in exit.
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
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, 0
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, 0
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, 0
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, 0
; CHECK:         v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; CHECK:       .LBB0_1:                                ; %loop.body
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:         v_mfma_f32_4x4x2bf16 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK:       ; %bb.2:                                ; %exit
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
; CHECK:         v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    ptr addrspace(1) %out,
    <2 x i16> %a,
    <2 x i16> %b,
    i32 %n) #0 {
entry:
  br label %loop.body

loop.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  ; 8 loop-carried <32 x float>: 8*32=256 VGPRs non-MFMA pressure.
  %c0 = phi <32 x float> [ zeroinitializer, %entry ], [ %c0n, %loop.body ]
  %c1 = phi <32 x float> [ zeroinitializer, %entry ], [ %c1n, %loop.body ]
  %c2 = phi <32 x float> [ zeroinitializer, %entry ], [ %c2n, %loop.body ]
  %c3 = phi <32 x float> [ zeroinitializer, %entry ], [ %c3n, %loop.body ]
  %c4 = phi <32 x float> [ zeroinitializer, %entry ], [ %c4n, %loop.body ]
  %c5 = phi <32 x float> [ zeroinitializer, %entry ], [ %c5n, %loop.body ]
  %c6 = phi <32 x float> [ zeroinitializer, %entry ], [ %c6n, %loop.body ]
  %c7 = phi <32 x float> [ zeroinitializer, %entry ], [ %c7n, %loop.body ]
  ; 9 MFMA accumulators: 9*4=36 VGPRs. Case 3: entry zeroinitializer -> AGPR.
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %r0, %loop.body ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %r1, %loop.body ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %r2, %loop.body ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %r3, %loop.body ]
  %acc4 = phi <4 x float> [ zeroinitializer, %entry ], [ %r4, %loop.body ]
  %acc5 = phi <4 x float> [ zeroinitializer, %entry ], [ %r5, %loop.body ]
  %acc6 = phi <4 x float> [ zeroinitializer, %entry ], [ %r6, %loop.body ]
  %acc7 = phi <4 x float> [ zeroinitializer, %entry ], [ %r7, %loop.body ]
  %acc8 = phi <4 x float> [ zeroinitializer, %entry ], [ %r8, %loop.body ]

  ; 9 MFMAs — results only used in exit (Case 2: cross-block non-MAI use).
  %r0 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc0, i32 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc1, i32 0, i32 0, i32 0)
  %r2 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc2, i32 0, i32 0, i32 0)
  %r3 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc3, i32 0, i32 0, i32 0)
  %r4 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc4, i32 0, i32 0, i32 0)
  %r5 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc5, i32 0, i32 0, i32 0)
  %r6 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc6, i32 0, i32 0, i32 0)
  %r7 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc7, i32 0, i32 0, i32 0)
  %r8 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %acc8, i32 0, i32 0, i32 0)

  ; Variable index prevents folding: keeps carriers live across all MFMAs.
  %eidx = and i32 %i, 31
  %c0e = extractelement <32 x float> %c0, i32 0
  %c1e = extractelement <32 x float> %c1, i32 0
  %c2e = extractelement <32 x float> %c2, i32 0
  %c3e = extractelement <32 x float> %c3, i32 0
  %c4e = extractelement <32 x float> %c4, i32 0
  %c5e = extractelement <32 x float> %c5, i32 0
  %c6e = extractelement <32 x float> %c6, i32 0
  %c7e = extractelement <32 x float> %c7, i32 0
  %c0n = insertelement <32 x float> %c0, float %c0e, i32 %eidx
  %c1n = insertelement <32 x float> %c1, float %c1e, i32 %eidx
  %c2n = insertelement <32 x float> %c2, float %c2e, i32 %eidx
  %c3n = insertelement <32 x float> %c3, float %c3e, i32 %eidx
  %c4n = insertelement <32 x float> %c4, float %c4e, i32 %eidx
  %c5n = insertelement <32 x float> %c5, float %c5e, i32 %eidx
  %c6n = insertelement <32 x float> %c6, float %c6e, i32 %eidx
  %c7n = insertelement <32 x float> %c7, float %c7e, i32 %eidx

  %csum = fadd float %c0e, %c1e
  store float %csum, ptr addrspace(1) %out, align 4

  %i.next = add i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop.body

exit:
  ; Case 2: non-MAI uses of MFMA dst in exit block (different block from loop.body).
  ; Case 3: reaching def of each %acc in entry is zeroinitializer (non-MAI).
  ; Expected: v_accvgpr_read (AGPR->VGPR) inserted before these extractelements.
  %e0 = extractelement <4 x float> %r0, i32 0
  %e1 = extractelement <4 x float> %r1, i32 0
  store float %e0, ptr addrspace(1) %out, align 4
  %p1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %e1, ptr addrspace(1) %p1, align 4
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }
