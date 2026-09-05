; NOTE: Do not autogenerate. This test intentionally checks only resource usage.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck %s

; Regression test for https://github.com/llvm/llvm-project/issues/219377#issuecomment-5510810499.
; The loop's original pressure is 45 VGPRs. With an 8-register allocation
; granule, alignTo(45 + 16 + 3, 8) equals the 64-register occupancy boundary.
; Reserving that final granule stops pending nodes from extending live ranges.

; CHECK-LABEL: avoid_unified_vgpr_occupancy_cliff:
; CHECK:      ; NumVgprs: 64{{$}}
; CHECK-NEXT: ; NumAgprs: 0{{$}}
; CHECK-NEXT: ; TotalNumVgprs: 64{{$}}
; CHECK-NEXT: ; ScratchSize: 0{{$}}
; CHECK:      ; Occupancy: 8{{$}}

define amdgpu_kernel void @avoid_unified_vgpr_occupancy_cliff(<8 x bfloat> %matrix0, <8 x bfloat> %matrix1, <8 x bfloat> %matrix2) {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %acc0 = phi float [ 0.000000e+00, %entry ], [ %next.acc0, %loop ]
  %acc1 = phi float [ 0.000000e+00, %entry ], [ %next.acc1, %loop ]
  %acc2 = phi float [ 0.000000e+00, %entry ], [ %next.acc2, %loop ]
  %acc3 = phi float [ 0.000000e+00, %entry ], [ %next.acc3, %loop ]
  %acc0.insert0 = insertelement <4 x float> zeroinitializer, float %acc0, i64 0
  %acc0.insert1 = insertelement <4 x float> %acc0.insert0, float %acc0, i64 1
  %acc0.seed = insertelement <4 x float> %acc0.insert1, float 0.000000e+00, i64 0
  %chain0.0 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %acc0.seed, i32 0, i32 0, i32 0)
  %acc1.insert0 = insertelement <4 x float> zeroinitializer, float %acc1, i64 0
  %acc1.insert1 = insertelement <4 x float> %acc1.insert0, float %acc1, i64 1
  %acc1.seed = insertelement <4 x float> %acc1.insert1, float 0.000000e+00, i64 0
  %chain1.0 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %matrix0, <8 x bfloat> zeroinitializer, <4 x float> %acc1.seed, i32 0, i32 0, i32 0)
  %acc2.insert0 = insertelement <4 x float> zeroinitializer, float %acc2, i64 0
  %acc2.seed = insertelement <4 x float> %acc2.insert0, float %acc2, i64 0
  %acc3.insert0 = insertelement <4 x float> zeroinitializer, float %acc3, i64 0
  %acc3.seed = insertelement <4 x float> %acc3.insert0, float %acc3, i64 0
  %chain3.0 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %matrix1, <8 x bfloat> zeroinitializer, <4 x float> %acc3.seed, i32 0, i32 0, i32 0)
  %chain0.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.0, i32 0, i32 0, i32 0)
  %chain1.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %matrix2, <8 x bfloat> zeroinitializer, <4 x float> %chain1.0, i32 0, i32 0, i32 0)
  %chain2.0 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> <bfloat 0.000000e+00, bfloat 0.000000e+00, bfloat 0.000000e+00, bfloat 0.000000e+00, bfloat 1.000000e+00, bfloat 1.000000e+00, bfloat 1.000000e+00, bfloat 1.000000e+00>, <8 x bfloat> zeroinitializer, <4 x float> %acc2.seed, i32 0, i32 0, i32 0)
  %chain2.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer, <4 x float> %chain2.0, i32 0, i32 0, i32 0)
  ; Preserve the scheduling-region boundary and reserve low VGPRs so the
  ; allocation difference straddles the 64-register occupancy boundary.
  tail call void asm sideeffect "", "~{v[0:15]},~{v[16:23]}"()
  %chain0.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.1, i32 0, i32 0, i32 0)
  %chain0.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.2, i32 0, i32 0, i32 0)
  %chain0.4 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.3, i32 0, i32 0, i32 0)
  %chain0.5 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.4, i32 0, i32 0, i32 0)
  %chain0.6 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.5, i32 0, i32 0, i32 0)
  %chain0.7 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain0.6, i32 0, i32 0, i32 0)
  %next.acc0 = extractelement <4 x float> %chain0.7, i64 0
  %chain1.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.1, i32 0, i32 0, i32 0)
  %chain1.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.2, i32 0, i32 0, i32 0)
  %chain1.4 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.3, i32 0, i32 0, i32 0)
  %chain1.5 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.4, i32 0, i32 0, i32 0)
  %chain1.6 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.5, i32 0, i32 0, i32 0)
  %chain1.7 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain1.6, i32 0, i32 0, i32 0)
  %next.acc1 = extractelement <4 x float> %chain1.7, i64 0
  %chain2.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.1, i32 0, i32 0, i32 0)
  %chain2.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.2, i32 0, i32 0, i32 0)
  %chain2.4 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.3, i32 0, i32 0, i32 0)
  %chain2.5 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.4, i32 0, i32 0, i32 0)
  %chain2.6 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.5, i32 0, i32 0, i32 0)
  %chain2.7 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain2.6, i32 0, i32 0, i32 0)
  %next.acc2 = extractelement <4 x float> %chain2.7, i64 0
  %chain3.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.0, i32 0, i32 0, i32 0)
  %chain3.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.1, i32 0, i32 0, i32 0)
  %chain3.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.2, i32 0, i32 0, i32 0)
  %chain3.4 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.3, i32 0, i32 0, i32 0)
  %chain3.5 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.4, i32 0, i32 0, i32 0)
  %chain3.6 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <4 x float> %chain3.5, i32 0, i32 0, i32 0)
  %next.acc3 = extractelement <4 x float> %chain3.6, i64 0
  br label %loop
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
