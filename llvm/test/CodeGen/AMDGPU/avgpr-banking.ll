; REQUIRES: asserts

; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-avgpr-banking=1 \
; RUN:     -debug-only=machine-scheduler %s -o - 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=BANK,NOSPILL \
; RUN:     -implicit-check-not="Region is known to spill" \
; RUN:     -implicit-check-not="Reverting scheduling"
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-avgpr-banking=0 \
; RUN:     -debug-only=machine-scheduler %s -o - 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=NOBANK,NOSPILL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-use-amdgpu-trackers=1 \
; RUN:     -amdgpu-avgpr-banking=1 -debug-only=machine-scheduler %s -o - 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=BANK,NOSPILL \
; RUN:     -implicit-check-not="Region is known to spill" \
; RUN:     -implicit-check-not="Reverting scheduling"
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-use-amdgpu-trackers=1 \
; RUN:     -amdgpu-avgpr-banking=0 -debug-only=machine-scheduler %s -o - 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=NOBANK,NOSPILL

; Nearly all of this kernel's vector pressure sits in av_* values, which the
; register allocator may place in either the ArchVGPR or the AGPR bank.
;
; The IR is shared with coexec-rewrite-mfma.ll, reduced there from a
; Triton-generated GEMM with 30 MFMAs in a phi-carried loop.

; BANK: Region register pressure: VGPRs: {{[0-9]+}} (AV: {{[0-9]+}}) AGPRs: {{[0-9]+}}
; NOBANK: Region register pressure: VGPRs: {{[0-9]+}} (AV: {{[0-9]+}}) AGPRs: {{[0-9]+}}

; The two stages report these in either order depending on the tracker in use.
; NOBANK-DAG: Region is known to spill
; NOBANK-DAG: Reverting scheduling for region

; The checks above are what the scheduler predicts. Neither configuration
; spills in the end.
; NOSPILL: ScratchSize: 0

define amdgpu_kernel void @mfma_loop_high_av_pressure() #0 {
entry:
  %i = load <8 x half>, ptr addrspace(3) null, align 16
  %i1 = shufflevector <8 x half> %i, <8 x half> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %i2 = shufflevector <8 x half> %i, <8 x half> zeroinitializer, <2 x i32> <i32 2, i32 3>
  br label %loop

loop:                                                ; preds = %loop, %entry
  %i3 = phi float [ 0.000000e+00, %entry ], [ %i52, %loop ]
  %i4 = phi float [ 0.000000e+00, %entry ], [ %i53, %loop ]
  %i5 = phi float [ 0.000000e+00, %entry ], [ %i60, %loop ]
  %i6 = phi float [ 0.000000e+00, %entry ], [ %i61, %loop ]
  %i7 = phi float [ 0.000000e+00, %entry ], [ %i62, %loop ]
  %i8 = phi float [ 0.000000e+00, %entry ], [ %i63, %loop ]
  %i9 = phi float [ 0.000000e+00, %entry ], [ %i70, %loop ]
  %i10 = phi float [ 0.000000e+00, %entry ], [ %i71, %loop ]
  %i11 = phi float [ 0.000000e+00, %entry ], [ %i72, %loop ]
  %i12 = phi float [ 0.000000e+00, %entry ], [ %i73, %loop ]
  %i13 = phi float [ 0.000000e+00, %entry ], [ %i79, %loop ]
  %i14 = phi float [ 0.000000e+00, %entry ], [ %i80, %loop ]
  %i15 = phi float [ 0.000000e+00, %entry ], [ %i81, %loop ]
  %i16 = phi float [ 0.000000e+00, %entry ], [ %i86, %loop ]
  %i17 = phi float [ 0.000000e+00, %entry ], [ %i87, %loop ]
  %i18 = phi float [ 0.000000e+00, %entry ], [ %i92, %loop ]
  %i19 = phi float [ 0.000000e+00, %entry ], [ %i93, %loop ]
  %i20 = phi float [ 0.000000e+00, %entry ], [ %i99, %loop ]
  %i21 = phi float [ 0.000000e+00, %entry ], [ %i100, %loop ]
  %i22 = phi float [ 0.000000e+00, %entry ], [ %i23, %loop ]
  %i23 = phi float [ 0.000000e+00, %entry ], [ %i105, %loop ]
  %i24 = phi float [ 0.000000e+00, %entry ], [ %i112, %loop ]
  %i25 = phi float [ 0.000000e+00, %entry ], [ %i113, %loop ]
  %i26 = phi float [ 0.000000e+00, %entry ], [ %i114, %loop ]
  %i27 = phi float [ 0.000000e+00, %entry ], [ %i115, %loop ]
  %i28 = phi float [ 0.000000e+00, %entry ], [ %i122, %loop ]
  %i29 = phi float [ 0.000000e+00, %entry ], [ %i123, %loop ]
  %i30 = phi float [ 0.000000e+00, %entry ], [ %i124, %loop ]
  %i31 = phi float [ 0.000000e+00, %entry ], [ %i125, %loop ]
  %i32 = phi float [ 0.000000e+00, %entry ], [ %i132, %loop ]
  %i33 = phi float [ 0.000000e+00, %entry ], [ %i133, %loop ]
  %i34 = phi float [ 0.000000e+00, %entry ], [ %i134, %loop ]
  %i35 = phi float [ 0.000000e+00, %entry ], [ %i135, %loop ]
  %i36 = phi float [ 0.000000e+00, %entry ], [ %i140, %loop ]
  %i37 = phi float [ 0.000000e+00, %entry ], [ %i141, %loop ]
  %i38 = phi float [ 0.000000e+00, %entry ], [ %i146, %loop ]
  %i39 = phi float [ 0.000000e+00, %entry ], [ %i147, %loop ]
  %i40 = phi float [ 0.000000e+00, %entry ], [ %i152, %loop ]
  %i41 = phi float [ 0.000000e+00, %entry ], [ %i40, %loop ]
  %p0 = phi float [ 0.000000e+00, %entry ], [ %e0, %loop ]
  %p1 = phi float [ 0.000000e+00, %entry ], [ %e1, %loop ]
  %i42 = phi <2 x half> [ %i1, %entry ], [ zeroinitializer, %loop ]
  %i43 = phi <2 x half> [ %i2, %entry ], [ zeroinitializer, %loop ]
  %i44 = phi <8 x half> [ zeroinitializer, %entry ], [ <half 0.000000e+00, half 0.000000e+00, half poison, half poison, half poison, half poison, half poison, half poison>, %loop ]
  %i45 = shufflevector <2 x half> %i42, <2 x half> %i43, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %i46 = shufflevector <8 x half> %i45, <8 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %i47 = shufflevector <8 x half> %i45, <8 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %i48 = insertelement <4 x float> zeroinitializer, float %i3, i64 2
  %i49 = insertelement <4 x float> %i48, float %i4, i64 3
  %i50 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %i46, <8 x half> zeroinitializer, <4 x float> %i49, i32 0, i32 0, i32 0)
  %i51 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i50, i32 0, i32 0, i32 0)
  %i52 = extractelement <4 x float> %i51, i64 2
  %i53 = extractelement <4 x float> %i51, i64 3
  %i54 = insertelement <4 x float> zeroinitializer, float %i5, i64 0
  %i55 = insertelement <4 x float> %i54, float %i6, i64 1
  %i56 = insertelement <4 x float> %i55, float %i7, i64 2
  %i57 = insertelement <4 x float> %i56, float %i8, i64 3
  %i58 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i57, i32 0, i32 0, i32 0)
  %i59 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i58, i32 0, i32 0, i32 0)
  %i60 = extractelement <4 x float> %i59, i64 0
  %i61 = extractelement <4 x float> %i59, i64 1
  %i62 = extractelement <4 x float> %i59, i64 2
  %i63 = extractelement <4 x float> %i59, i64 3
  %i64 = insertelement <4 x float> zeroinitializer, float %i9, i64 0
  %i65 = insertelement <4 x float> %i64, float %i10, i64 1
  %i66 = insertelement <4 x float> %i65, float %i11, i64 2
  %i67 = insertelement <4 x float> %i66, float %i12, i64 3
  %i68 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i67, i32 0, i32 0, i32 0)
  %i69 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i68, i32 0, i32 0, i32 0)
  %i70 = extractelement <4 x float> %i69, i64 0
  %i71 = extractelement <4 x float> %i69, i64 1
  %i72 = extractelement <4 x float> %i69, i64 2
  %i73 = extractelement <4 x float> %i69, i64 3
  %i74 = insertelement <4 x float> zeroinitializer, float %i13, i64 0
  %i75 = insertelement <4 x float> %i74, float %i14, i64 1
  %i76 = insertelement <4 x float> %i75, float %i15, i64 2
  %i77 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i76, i32 0, i32 0, i32 0)
  %i78 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i77, i32 0, i32 0, i32 0)
  %i79 = extractelement <4 x float> %i78, i64 0
  %i80 = extractelement <4 x float> %i78, i64 1
  %i81 = extractelement <4 x float> %i78, i64 2
  %i82 = insertelement <4 x float> zeroinitializer, float %i16, i64 2
  %i83 = insertelement <4 x float> %i82, float %i17, i64 3
  %i84 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i83, i32 0, i32 0, i32 0)
  %i85 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i84, i32 0, i32 0, i32 0)
  %i86 = extractelement <4 x float> %i85, i64 2
  %i87 = extractelement <4 x float> %i85, i64 3
  %i88 = insertelement <4 x float> zeroinitializer, float %i18, i64 0
  %i89 = insertelement <4 x float> %i88, float %i19, i64 1
  %i90 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i89, i32 0, i32 0, i32 0)
  %i91 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i90, i32 0, i32 0, i32 0)
  %i92 = extractelement <4 x float> %i91, i64 0
  %i93 = extractelement <4 x float> %i91, i64 1
  %i94 = shufflevector <8 x half> zeroinitializer, <8 x half> %i44, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %i95 = insertelement <4 x float> zeroinitializer, float %i20, i64 0
  %i96 = insertelement <4 x float> %i95, float %i21, i64 1
  %i97 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %i94, <8 x half> zeroinitializer, <4 x float> %i96, i32 0, i32 0, i32 0)
  %i98 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i97, i32 0, i32 0, i32 0)
  %i99 = extractelement <4 x float> %i98, i64 0
  %i100 = extractelement <4 x float> %i98, i64 1
  %i101 = insertelement <4 x float> poison, float %i22, i64 0
  %i102 = insertelement <4 x float> %i101, float %i23, i64 0
  %i103 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i102, i32 0, i32 0, i32 0)
  %i104 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i103, i32 0, i32 0, i32 0)
  %i105 = extractelement <4 x float> %i104, i64 0
  %i106 = insertelement <4 x float> zeroinitializer, float %i24, i64 0
  %i107 = insertelement <4 x float> %i106, float %i25, i64 1
  %i108 = insertelement <4 x float> %i107, float %i26, i64 2
  %i109 = insertelement <4 x float> %i108, float %i27, i64 3
  %i110 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %i47, <8 x half> zeroinitializer, <4 x float> %i109, i32 0, i32 0, i32 0)
  %i111 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i110, i32 0, i32 0, i32 0)
  %i112 = extractelement <4 x float> %i111, i64 0
  %i113 = extractelement <4 x float> %i111, i64 1
  %i114 = extractelement <4 x float> %i111, i64 2
  %i115 = extractelement <4 x float> %i111, i64 3
  %i116 = insertelement <4 x float> zeroinitializer, float %i28, i64 0
  %i117 = insertelement <4 x float> %i116, float %i29, i64 1
  %i118 = insertelement <4 x float> %i117, float %i30, i64 2
  %i119 = insertelement <4 x float> %i118, float %i31, i64 3
  %i120 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i119, i32 0, i32 0, i32 0)
  %i121 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i120, i32 0, i32 0, i32 0)
  %i122 = extractelement <4 x float> %i121, i64 0
  %i123 = extractelement <4 x float> %i121, i64 1
  %i124 = extractelement <4 x float> %i121, i64 2
  %i125 = extractelement <4 x float> %i121, i64 3
  %i126 = insertelement <4 x float> zeroinitializer, float %i32, i64 0
  %i127 = insertelement <4 x float> %i126, float %i33, i64 1
  %i128 = insertelement <4 x float> %i127, float %i34, i64 2
  %i129 = insertelement <4 x float> %i128, float %i35, i64 3
  %i130 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i129, i32 0, i32 0, i32 0)
  %i131 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i130, i32 0, i32 0, i32 0)
  %i132 = extractelement <4 x float> %i131, i64 0
  %i133 = extractelement <4 x float> %i131, i64 1
  %i134 = extractelement <4 x float> %i131, i64 2
  %i135 = extractelement <4 x float> %i131, i64 3
  %i136 = insertelement <4 x float> zeroinitializer, float %i36, i64 0
  %i137 = insertelement <4 x float> %i136, float %i37, i64 1
  %i138 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i137, i32 0, i32 0, i32 0)
  %i139 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i138, i32 0, i32 0, i32 0)
  %i140 = extractelement <4 x float> %i139, i64 0
  %i141 = extractelement <4 x float> %i139, i64 1
  %i142 = insertelement <4 x float> zeroinitializer, float %i38, i64 0
  %i143 = insertelement <4 x float> %i142, float %i39, i64 1
  %i144 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i143, i32 0, i32 0, i32 0)
  %i145 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i144, i32 0, i32 0, i32 0)
  %i146 = extractelement <4 x float> %i145, i64 0
  %i147 = extractelement <4 x float> %i145, i64 1
  %i148 = insertelement <4 x float> <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %i40, i64 2
  %i149 = insertelement <4 x float> %i148, float %i41, i64 3
  %i150 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i149, i32 0, i32 0, i32 0)
  %i151 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %i150, i32 0, i32 0, i32 0)
  %i152 = extractelement <4 x float> %i151, i64 0
  %x0 = insertelement <4 x float> zeroinitializer, float %p0, i64 0
  %x1 = insertelement <4 x float> %x0, float %p1, i64 1
  %x2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %x1, i32 0, i32 0, i32 0)
  %x3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %x2, i32 0, i32 0, i32 0)
  %e0 = extractelement <4 x float> %x3, i64 0
  %e1 = extractelement <4 x float> %x3, i64 1
  br label %loop
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
